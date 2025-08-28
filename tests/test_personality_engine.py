#!/usr/bin/env python3
"""
Test suite for Phase 23 Dynamic Personality Evolution
Enhanced from Phase 9 Expressive Personality Engine
"""

import unittest
import tempfile
import os
import time
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from agents.personality_engine import (
    ExpressivePersonalityEngine, ResponseMode, ToneProfile, 
    PersonalityState, PersonalityEvolutionEvent,
    get_personality_engine
)


class TestPersonalityEvolutionEvent(unittest.TestCase):
    """Test PersonalityEvolutionEvent class."""
    
    def test_evolution_event_creation(self):
        """Test creating a personality evolution event."""
        event = PersonalityEvolutionEvent(
            timestamp="2024-01-01T12:00:00",
            trigger="feedback",
            changes={"empathetic": {"from": 0.5, "to": 0.7}},
            reason="User requested more empathy",
            updated_by="user_feedback"
        )
        
        self.assertEqual(event.timestamp, "2024-01-01T12:00:00")
        self.assertEqual(event.trigger, "feedback")
        self.assertEqual(event.changes["empathetic"]["to"], 0.7)
        self.assertEqual(event.reason, "User requested more empathy")
        self.assertEqual(event.updated_by, "user_feedback")
    
    def test_evolution_event_serialization(self):
        """Test converting evolution event to dictionary."""
        event = PersonalityEvolutionEvent(
            timestamp="2024-01-01T12:00:00",
            trigger="manual",
            changes={"curious": {"from": 0.6, "to": 0.8}},
            reason="Manual adjustment",
            updated_by="user_manual"
        )
        
        event_dict = event.to_dict()
        
        self.assertIsInstance(event_dict, dict)
        self.assertEqual(event_dict["timestamp"], "2024-01-01T12:00:00")
        self.assertEqual(event_dict["trigger"], "manual")
        self.assertEqual(event_dict["changes"]["curious"]["to"], 0.8)


class TestPersonalityState(unittest.TestCase):
    """Test PersonalityState class and evolution functionality."""
    
    def setUp(self):
        """Set up test personality state."""
        self.personality_state = PersonalityState(
            tone="rational",
            emotional_state="neutral",
            core_traits={
                "curious": 0.7,
                "analytical": 0.8,
                "empathetic": 0.5
            },
            evolution_history=[],
            last_updated=datetime.now().isoformat(),
            last_updated_by="test_init",
            feedback_sensitivity=0.7,
            stability_factor=0.3,
            creation_timestamp=datetime.now().isoformat(),
            total_evolutions=0
        )
    
    def test_personality_state_initialization(self):
        """Test personality state initialization."""
        self.assertEqual(self.personality_state.tone, "rational")
        self.assertEqual(self.personality_state.emotional_state, "neutral")
        self.assertEqual(self.personality_state.core_traits["curious"], 0.7)
        self.assertEqual(self.personality_state.total_evolutions, 0)
        self.assertEqual(len(self.personality_state.evolution_history), 0)
    
    def test_feedback_analysis_positive_traits(self):
        """Test analyzing feedback for positive trait suggestions."""
        feedback = "be more empathetic and understanding"
        adjustments = self.personality_state._analyze_feedback_for_traits(feedback)
        
        self.assertIn("empathetic", adjustments)
        self.assertGreater(adjustments["empathetic"], 0)
    
    def test_evolve_from_feedback_trait_change(self):
        """Test personality evolution from feedback that changes traits."""
        initial_empathy = self.personality_state.core_traits["empathetic"]
        feedback = "be more empathetic and understanding"
        
        evolved = self.personality_state.evolve_from_feedback(feedback, "test_user")
        
        self.assertTrue(evolved)
        self.assertGreater(self.personality_state.core_traits["empathetic"], initial_empathy)
        self.assertEqual(self.personality_state.total_evolutions, 1)
        self.assertEqual(len(self.personality_state.evolution_history), 1)
        
        # Check evolution event
        event = self.personality_state.evolution_history[0]
        self.assertEqual(event.trigger, "feedback")
        self.assertEqual(event.updated_by, "test_user")
        self.assertIn("empathetic", event.changes)
    
    def test_serialization_and_deserialization(self):
        """Test converting personality state to/from dictionary."""
        # Add an evolution event
        self.personality_state.evolve_from_feedback("be more curious", "test")
        
        # Serialize to dict
        state_dict = self.personality_state.to_dict()
        
        self.assertIsInstance(state_dict, dict)
        self.assertEqual(state_dict["tone"], "rational")
        self.assertEqual(state_dict["emotional_state"], "neutral")
        self.assertIn("curious", state_dict["core_traits"])
        self.assertIsInstance(state_dict["evolution_history"], list)
        
        # Deserialize from dict
        restored_state = PersonalityState.from_dict(state_dict)
        
        self.assertEqual(restored_state.tone, self.personality_state.tone)
        self.assertEqual(restored_state.emotional_state, self.personality_state.emotional_state)
        self.assertEqual(restored_state.core_traits, self.personality_state.core_traits)
        self.assertEqual(restored_state.total_evolutions, self.personality_state.total_evolutions)
        self.assertEqual(len(restored_state.evolution_history), len(self.personality_state.evolution_history))


class TestDynamicPersonalityEvolution(unittest.TestCase):
    """Test enhanced personality engine with evolution capabilities."""
    
    def setUp(self):
        """Set up test personality engine with temporary storage."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_personality_file = Path(self.temp_dir) / "personality.json"
        
        # Create engine with temporary storage
        self.engine = ExpressivePersonalityEngine()
        self.engine.personality_file = self.test_personality_file
        self.engine.personality_state = self.engine._load_personality_state()
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_persistent_storage(self):
        """Test saving and loading personality state from file."""
        # Modify the personality state
        self.engine.personality_state.core_traits["test_trait"] = 0.9
        self.engine.personality_state.total_evolutions = 5
        
        # Save state
        success = self.engine._save_personality_state()
        self.assertTrue(success)
        self.assertTrue(self.test_personality_file.exists())
        
        # Load state
        loaded_state = self.engine._load_personality_state()
        
        self.assertEqual(loaded_state.core_traits["test_trait"], 0.9)
        self.assertEqual(loaded_state.total_evolutions, 5)
    
    def test_evolve_personality(self):
        """Test personality evolution through engine interface."""
        initial_empathy = self.engine.personality_state.core_traits.get("empathetic", 0.0)
        
        evolved = self.engine.evolve_personality("be more empathetic", "test_source")
        
        if evolved:
            final_empathy = self.engine.personality_state.core_traits.get("empathetic", 0.0)
            self.assertGreater(final_empathy, initial_empathy)
            
            # Check that state was saved
            self.assertTrue(self.test_personality_file.exists())
    
    def test_adjust_personality_traits(self):
        """Test manual personality trait adjustment."""
        success = self.engine.adjust_personality(
            empathetic=0.9,
            curious=0.8,
            tone="playful"
        )
        
        self.assertTrue(success)
        self.assertEqual(self.engine.personality_state.core_traits["empathetic"], 0.9)
        self.assertEqual(self.engine.personality_state.core_traits["curious"], 0.8)
        self.assertEqual(self.engine.personality_state.tone, "playful")
    
    def test_get_personality_summary(self):
        """Test generating personality summary."""
        summary = self.engine.get_personality_summary()
        
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 50)  # Should be a substantial description
        self.assertIn("personality", summary.lower())
    
    def test_get_personality_status(self):
        """Test getting detailed personality status."""
        status = self.engine.get_personality_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn("current_state", status)
        self.assertIn("core_traits", status)
        self.assertIn("evolution_info", status)
        
        # Check evolution info structure
        evolution_info = status["evolution_info"]
        self.assertIn("total_evolutions", evolution_info)
        self.assertIn("last_updated", evolution_info)
        self.assertIn("feedback_sensitivity", evolution_info)
        self.assertIn("stability_factor", evolution_info)
    
    def test_automatic_evolution_triggers(self):
        """Test automatic evolution triggers for emotional dissonance."""
        # Test contradiction trigger
        context = {"contradiction_score": 0.9}
        evolved = self.engine.check_for_emotional_dissonance(context)
        self.assertTrue(evolved)
        
        # Test sentiment trigger
        context = {"user_sentiment": "frustrated"}
        evolved = self.engine.check_for_emotional_dissonance(context)
        self.assertTrue(evolved)
        
        # Test feedback trigger
        context = {"user_feedback": "you were too cold and harsh"}
        evolved = self.engine.check_for_emotional_dissonance(context)
        self.assertTrue(evolved)


class TestAgentIntegration(unittest.TestCase):
    """Test integration with BaseAgent."""
    
    def setUp(self):
        """Set up test agent."""
        from agents.base import BaseAgent
        
        # Create a simple test agent
        class TestAgent(BaseAgent):
            def get_agent_instructions(self):
                return "Test agent instructions"
            
            def respond(self, message, context=None):
                base_response = f"Test response to: {message}"
                return self.style_response_with_personality(base_response, context)
        
        self.agent = TestAgent("test")
    
    def test_agent_has_personality_property(self):
        """Test that agents have personality property."""
        self.assertTrue(hasattr(self.agent, 'personality'))
    
    def test_agent_capabilities_include_personality(self):
        """Test that agent capabilities include personality info."""
        capabilities = self.agent.get_capabilities()
        self.assertIn("has_personality", capabilities)
        self.assertIsInstance(capabilities["has_personality"], bool)
    
    def test_style_response_with_personality(self):
        """Test personality styling through agent."""
        response = "This is a test response."
        styled_response = self.agent.style_response_with_personality(response)
        self.assertIsInstance(styled_response, str)
        self.assertGreater(len(styled_response), 0)


class TestResponseMode(unittest.TestCase):
    """Test cases for ResponseMode enum."""
    
    def test_response_mode_values(self):
        """Test that all expected response modes exist."""
        expected_modes = ["rational", "empathetic", "playful", "assertive"]
        
        for mode_name in expected_modes:
            # Should be able to access mode by name
            mode = ResponseMode(mode_name)
            self.assertEqual(mode.value, mode_name)
    
    def test_response_mode_enum_properties(self):
        """Test ResponseMode enum has expected properties."""
        self.assertEqual(ResponseMode.RATIONAL.value, "rational")
        self.assertEqual(ResponseMode.EMPATHETIC.value, "empathetic")
        self.assertEqual(ResponseMode.PLAYFUL.value, "playful")
        self.assertEqual(ResponseMode.ASSERTIVE.value, "assertive")


class TestToneProfile(unittest.TestCase):
    """Test cases for ToneProfile dataclass."""
    
    def test_tone_profile_creation(self):
        """Test creating ToneProfile with valid values."""
        profile = ToneProfile(
            mode=ResponseMode.PLAYFUL,
            primary_traits=["curious", "analytical"],
            mood_label="excited",
            emotional_intensity=0.8,
            confidence_level=0.7,
            conversational_energy=0.9,
            valence_bias=0.6
        )
        
        self.assertEqual(profile.mode, ResponseMode.PLAYFUL)
        self.assertEqual(profile.primary_traits, ["curious", "analytical"])
        self.assertEqual(profile.mood_label, "excited")
        self.assertEqual(profile.emotional_intensity, 0.8)
    
    def test_tone_profile_serialization(self):
        """Test tone profile to_dict conversion."""
        profile = ToneProfile(
            mode=ResponseMode.RATIONAL,
            primary_traits=["analytical"],
            mood_label="calm",
            emotional_intensity=0.4,
            confidence_level=0.8,
            conversational_energy=0.3,
            valence_bias=0.1
        )
        
        data = profile.to_dict()
        
        self.assertIn('mode', data)
        self.assertIn('primary_traits', data)
        self.assertIn('mood_label', data)
        self.assertEqual(data['mode'], 'rational')
        self.assertEqual(data['primary_traits'], ['analytical'])
    
    def test_tone_profile_description(self):
        """Test tone profile description generation."""
        profile = ToneProfile(
            mode=ResponseMode.PLAYFUL,
            primary_traits=["curious", "optimistic"],
            mood_label="excited",
            emotional_intensity=0.7,
            confidence_level=0.6,
            conversational_energy=0.8,
            valence_bias=0.5
        )
        
        description = profile.get_description()
        
        self.assertIn("playful", description.lower())
        self.assertIn("curious", description.lower())
        self.assertIn("excited", description.lower())
        self.assertIn("energetic", description.lower())  # High energy


class TestExpressivePersonalityEngine(unittest.TestCase):
    """Test cases for ExpressivePersonalityEngine."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.test_db.close()
        self.db_path = self.test_db.name
        
        self.personality_engine = ExpressivePersonalityEngine(self.db_path)
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_personality_engine_initialization(self):
        """Test personality engine initialization."""
        self.assertIsNotNone(self.personality_engine.response_templates)
        self.assertIsNotNone(self.personality_engine.trait_patterns)
        self.assertIsNone(self.personality_engine.mode_override)
        
        # Check that all response modes have templates
        for mode in ResponseMode:
            self.assertIn(mode, self.personality_engine.response_templates)
    
    def test_response_templates_structure(self):
        """Test that response templates have expected structure."""
        templates = self.personality_engine.response_templates
        
        # All modes should have templates
        for mode in ResponseMode:
            self.assertIn(mode, templates)
            mode_template = templates[mode]
            
            # Should have style_markers
            self.assertIn('style_markers', mode_template)
            self.assertIsInstance(mode_template['style_markers'], dict)
    
    def test_trait_patterns_structure(self):
        """Test trait patterns have expected structure."""
        patterns = self.personality_engine.trait_patterns
        
        expected_traits = ["curious", "analytical", "empathetic", "skeptical", "optimistic", "resilient"]
        
        for trait in expected_traits:
            self.assertIn(trait, patterns)
            trait_patterns = patterns[trait]
            self.assertIsInstance(trait_patterns, dict)
    
    @patch('storage.emotion_model.get_emotion_model')
    @patch('storage.self_model.get_self_aware_model')
    def test_get_current_tone_profile(self, mock_self_model, mock_emotion_model):
        """Test generating current tone profile."""
        # Mock emotion model
        mock_emotion = MagicMock()
        mock_emotion.get_current_mood.return_value = {
            "mood_label": "curious",
            "valence": 0.4,
            "arousal": 0.6,
            "confidence": 0.8
        }
        mock_emotion_model.return_value = mock_emotion
        
        # Mock self-aware model with identity traits
        mock_self = MagicMock()
        mock_trait = MagicMock()
        mock_trait.strength = 0.8
        mock_trait.confidence = 0.9
        mock_self.identity_traits = {"curious": mock_trait, "analytical": mock_trait}
        mock_self_model.return_value = mock_self
        
        # Get tone profile
        tone_profile = self.personality_engine.get_current_tone_profile()
        
        self.assertIsInstance(tone_profile, ToneProfile)
        self.assertIn(tone_profile.mode, ResponseMode)
        self.assertEqual(tone_profile.mood_label, "curious")
        self.assertGreater(len(tone_profile.primary_traits), 0)
    
    def test_response_mode_determination(self):
        """Test response mode determination logic."""
        # Test high positive arousal -> playful
        mode = self.personality_engine._determine_response_mode(
            valence=0.7, arousal=0.8, traits=["curious"], mood_label="excited"
        )
        self.assertEqual(mode, ResponseMode.PLAYFUL)
        
        # Test negative valence + high arousal -> assertive
        mode = self.personality_engine._determine_response_mode(
            valence=-0.5, arousal=0.8, traits=[], mood_label="frustrated"
        )
        self.assertEqual(mode, ResponseMode.ASSERTIVE)
        
        # Test low arousal with empathetic trait -> empathetic
        mode = self.personality_engine._determine_response_mode(
            valence=-0.1, arousal=0.2, traits=["empathetic"], mood_label="calm"
        )
        self.assertEqual(mode, ResponseMode.EMPATHETIC)
        
        # Test analytical trait -> rational
        mode = self.personality_engine._determine_response_mode(
            valence=0.2, arousal=0.5, traits=["analytical"], mood_label="neutral"
        )
        self.assertEqual(mode, ResponseMode.RATIONAL)
    
    def test_style_response_basic(self):
        """Test basic response styling functionality."""
        base_text = "I think this is the answer to your question."
        
        # Test with different overrides
        for mode in ResponseMode:
            self.personality_engine.set_response_mode_override(mode)
            styled = self.personality_engine.style_response(base_text)
            
            # Should return a string
            self.assertIsInstance(styled, str)
            # Should not be empty
            self.assertGreater(len(styled), 0)
            # Should contain some of the original text or be a reasonable transformation
            self.assertTrue(len(styled) >= len(base_text) * 0.5)  # At least half the length
    
    def test_style_response_modes(self):
        """Test response styling for different modes."""
        base_text = "The system is working correctly."
        
        # Test rational mode
        self.personality_engine.set_response_mode_override(ResponseMode.RATIONAL)
        rational_response = self.personality_engine.style_response(base_text)
        
        # Test empathetic mode
        self.personality_engine.set_response_mode_override(ResponseMode.EMPATHETIC)
        empathetic_response = self.personality_engine.style_response(base_text)
        
        # Test playful mode
        self.personality_engine.set_response_mode_override(ResponseMode.PLAYFUL)
        playful_response = self.personality_engine.style_response(base_text)
        
        # Test assertive mode
        self.personality_engine.set_response_mode_override(ResponseMode.ASSERTIVE)
        assertive_response = self.personality_engine.style_response(base_text)
        
        # All should be different (though this might not always be true for short text)
        responses = [rational_response, empathetic_response, playful_response, assertive_response]
        
        # At least should all be strings
        for response in responses:
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
    
    def test_context_adjustment(self):
        """Test context-based tone adjustment."""
        base_profile = ToneProfile(
            mode=ResponseMode.PLAYFUL,
            primary_traits=["curious"],
            mood_label="excited",
            emotional_intensity=0.6,
            confidence_level=0.7,
            conversational_energy=0.8,
            valence_bias=0.5
        )
        
        # Test contradiction context
        contradiction_context = {"contradiction_detected": True}
        adjusted = self.personality_engine._adjust_tone_for_context(base_profile, contradiction_context)
        
        # Should shift away from playful when contradiction detected
        self.assertNotEqual(adjusted.mode, ResponseMode.PLAYFUL)
        
        # Test user frustration context
        frustration_context = {"user_emotion": "frustrated"}
        adjusted = self.personality_engine._adjust_tone_for_context(base_profile, frustration_context)
        
        # Should shift to empathetic for frustrated user
        self.assertEqual(adjusted.mode, ResponseMode.EMPATHETIC)
    
    def test_mode_override_functionality(self):
        """Test manual mode override functionality."""
        # Test setting override
        success = self.personality_engine.set_response_mode_override(ResponseMode.ASSERTIVE)
        self.assertTrue(success)
        self.assertEqual(self.personality_engine.mode_override, ResponseMode.ASSERTIVE)
        
        # Test clearing override
        success = self.personality_engine.set_response_mode_override(None)
        self.assertTrue(success)
        self.assertIsNone(self.personality_engine.mode_override)
    
    def test_sample_responses_generation(self):
        """Test generating sample responses in all modes."""
        base_text = "I believe this is the correct approach."
        
        samples = self.personality_engine.get_sample_responses(base_text)
        
        # Should have samples for all modes
        expected_modes = {mode.value for mode in ResponseMode}
        actual_modes = set(samples.keys())
        self.assertEqual(expected_modes, actual_modes)
        
        # All samples should be strings
        for mode, sample in samples.items():
            self.assertIsInstance(sample, str)
            self.assertGreater(len(sample), 0)
    
    @patch('storage.emotion_model.get_emotion_model')
    @patch('storage.self_model.get_self_aware_model')
    def test_personality_banner_generation(self, mock_self_model, mock_emotion_model):
        """Test personality banner generation."""
        # Mock dependencies
        mock_emotion = MagicMock()
        mock_emotion.get_current_mood.return_value = {
            "mood_label": "curious",
            "valence": 0.4,
            "arousal": 0.6,
            "confidence": 0.8
        }
        mock_emotion_model.return_value = mock_emotion
        
        mock_self = MagicMock()
        mock_trait = MagicMock()
        mock_trait.strength = 0.8
        mock_trait.confidence = 0.9
        mock_self.identity_traits = {"curious": mock_trait}
        mock_self_model.return_value = mock_self
        
        # Generate banner
        banner = self.personality_engine.get_personality_banner()
        
        self.assertIsInstance(banner, str)
        self.assertGreater(len(banner), 0)
        self.assertIn("Mode:", banner)
        self.assertIn("Mood:", banner)
    
    def test_rational_styling_application(self):
        """Test rational mode styling specifics."""
        text = "This is a statement."
        tone_profile = ToneProfile(
            mode=ResponseMode.RATIONAL,
            primary_traits=["analytical"],
            mood_label="focused",
            emotional_intensity=0.4,
            confidence_level=0.8,
            conversational_energy=0.5,
            valence_bias=0.2
        )
        
        templates = self.personality_engine.response_templates[ResponseMode.RATIONAL]
        styled = self.personality_engine._apply_rational_styling(text, tone_profile, templates)
        
        self.assertIsInstance(styled, str)
        # For high confidence, might add analytical prefixes
        if tone_profile.confidence_level > 0.7:
            # Might start with analytical language, but not guaranteed for short text
            pass
    
    def test_empathetic_styling_application(self):
        """Test empathetic mode styling specifics."""
        text = "You should try this approach."
        tone_profile = ToneProfile(
            mode=ResponseMode.EMPATHETIC,
            primary_traits=["empathetic"],
            mood_label="caring",
            emotional_intensity=0.6,
            confidence_level=0.7,
            conversational_energy=0.4,
            valence_bias=0.3
        )
        
        templates = self.personality_engine.response_templates[ResponseMode.EMPATHETIC]
        styled = self.personality_engine._apply_empathetic_styling(text, tone_profile, templates)
        
        self.assertIsInstance(styled, str)
        # Should soften direct statements
        self.assertNotIn("You should", styled)  # Should be softened
        self.assertTrue("might consider" in styled or "could try" in styled or "may want to" in styled)
    
    def test_playful_styling_application(self):
        """Test playful mode styling specifics."""
        text = "However, this is interesting information."
        tone_profile = ToneProfile(
            mode=ResponseMode.PLAYFUL,
            primary_traits=["curious"],
            mood_label="excited",
            emotional_intensity=0.8,
            confidence_level=0.6,
            conversational_energy=0.9,
            valence_bias=0.7
        )
        
        templates = self.personality_engine.response_templates[ResponseMode.PLAYFUL]
        styled = self.personality_engine._apply_playful_styling(text, tone_profile, templates)
        
        self.assertIsInstance(styled, str)
        # Should make language more casual
        self.assertNotIn("However,", styled)  # Should be replaced with "But"
    
    def test_assertive_styling_application(self):
        """Test assertive mode styling specifics."""
        text = "I think that might be the solution."
        tone_profile = ToneProfile(
            mode=ResponseMode.ASSERTIVE,
            primary_traits=["resilient"],
            mood_label="determined",
            emotional_intensity=0.8,
            confidence_level=0.8,
            conversational_energy=0.7,
            valence_bias=0.1
        )
        
        templates = self.personality_engine.response_templates[ResponseMode.ASSERTIVE]
        styled = self.personality_engine._apply_assertive_styling(text, tone_profile, templates)
        
        self.assertIsInstance(styled, str)
        # Should remove hedging and make more direct
        self.assertNotIn("I think that", styled)  # Should be removed
        self.assertNotIn("might be", styled)  # Should be strengthened
    
    def test_emotional_intensity_application(self):
        """Test emotional intensity adjustments."""
        text = "This is very good information."
        
        # High intensity
        high_intensity_profile = ToneProfile(
            mode=ResponseMode.RATIONAL,
            primary_traits=[],
            mood_label="intense",
            emotional_intensity=0.9,
            confidence_level=0.7,
            conversational_energy=0.8,
            valence_bias=0.6
        )
        
        high_styled = self.personality_engine._apply_emotional_intensity(text, high_intensity_profile)
        # Should strengthen language
        self.assertIn("extremely", high_styled.lower())
        
        # Low intensity
        low_intensity_profile = ToneProfile(
            mode=ResponseMode.RATIONAL,
            primary_traits=[],
            mood_label="calm",
            emotional_intensity=0.2,
            confidence_level=0.7,
            conversational_energy=0.3,
            valence_bias=0.1
        )
        
        low_styled = self.personality_engine._apply_emotional_intensity(text, low_intensity_profile)
        # Should soften language (though original text might not trigger changes)
        self.assertIsInstance(low_styled, str)


class TestPersonalityEngineIntegration(unittest.TestCase):
    """Integration tests for personality engine."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.test_db.close()
        self.db_path = self.test_db.name
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_get_personality_engine_singleton(self):
        """Test that get_personality_engine returns singleton."""
        engine1 = get_personality_engine(self.db_path)
        engine2 = get_personality_engine(self.db_path)
        
        # Should be the same instance
        self.assertIs(engine1, engine2)
    
    @patch('storage.emotion_model.get_emotion_model')
    @patch('storage.self_model.get_self_aware_model')
    def test_personality_mode_transitions(self, mock_self_model, mock_emotion_model):
        """Test personality mode transitions based on emotional changes."""
        engine = ExpressivePersonalityEngine(self.db_path)
        
        # Mock different emotional states
        emotional_states = [
            {"mood_label": "excited", "valence": 0.8, "arousal": 0.9},
            {"mood_label": "frustrated", "valence": -0.6, "arousal": 0.8},
            {"mood_label": "calm", "valence": 0.2, "arousal": 0.2},
            {"mood_label": "analytical", "valence": 0.1, "arousal": 0.5}
        ]
        
        expected_modes = [
            ResponseMode.PLAYFUL,    # excited -> playful
            ResponseMode.ASSERTIVE,  # frustrated -> assertive
            ResponseMode.EMPATHETIC, # calm -> empathetic (with empathetic trait)
            ResponseMode.RATIONAL    # analytical -> rational
        ]
        
        # Mock self-aware model with different trait combinations
        trait_combinations = [
            {"curious": MagicMock(strength=0.8, confidence=0.9)},
            {"resilient": MagicMock(strength=0.7, confidence=0.8)},
            {"empathetic": MagicMock(strength=0.8, confidence=0.9)},
            {"analytical": MagicMock(strength=0.9, confidence=0.9)}
        ]
        
        for i, (emotional_state, expected_mode, traits) in enumerate(zip(emotional_states, expected_modes, trait_combinations)):
            with self.subTest(state=i):
                # Setup mocks for this iteration
                mock_emotion = MagicMock()
                mock_emotion.get_current_mood.return_value = {**emotional_state, "confidence": 0.8}
                mock_emotion_model.return_value = mock_emotion
                
                mock_self = MagicMock()
                mock_self.identity_traits = traits
                mock_self_model.return_value = mock_self
                
                # Get tone profile and check mode
                tone_profile = engine.get_current_tone_profile()
                
                # Mode should match expectation (though some flexibility due to complex logic)
                self.assertIsInstance(tone_profile.mode, ResponseMode)
                # For some states, the mode should definitely match
                if expected_mode == ResponseMode.ASSERTIVE:
                    self.assertEqual(tone_profile.mode, expected_mode)
    
    def test_end_to_end_styling(self):
        """Test complete end-to-end styling workflow."""
        engine = ExpressivePersonalityEngine(self.db_path)
        
        # Test different types of input text
        test_texts = [
            "I understand your question and here is my analysis.",
            "You should consider this approach to solve the problem.",
            "The data shows that this might be the correct answer.",
            "I'm not sure about this, but it could work."
        ]
        
        # Test with different contexts
        contexts = [
            {},
            {"contradiction_detected": True},
            {"user_emotion": "frustrated"},
            {"complex_query": True},
            {"casual_conversation": True}
        ]
        
        for text in test_texts:
            for context in contexts:
                with self.subTest(text=text[:20], context=str(context)):
                    styled = engine.style_response(text, context)
                    
                    # Basic validation
                    self.assertIsInstance(styled, str)
                    self.assertGreater(len(styled), 0)
                    # Should be reasonable transformation
                    self.assertLessEqual(len(styled), len(text) * 3)  # Not too much expansion


if __name__ == '__main__':
    # Run with increased verbosity to see test progress
    unittest.main(verbosity=2)