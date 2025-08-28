#!/usr/bin/env python3
"""
Test suite for Phase 8 Identity Signature functionality
"""

import unittest
import tempfile
import os
import time
from unittest.mock import patch, MagicMock

from storage.self_model import SelfAwareModel, IdentityTrait, get_self_aware_model


class TestIdentityTrait(unittest.TestCase):
    """Test cases for IdentityTrait dataclass."""
    
    def test_identity_trait_creation(self):
        """Test creating IdentityTrait with valid values."""
        trait = IdentityTrait(
            trait_name="curious",
            strength=0.7,
            confidence=0.8,
            evidence_count=5,
            last_updated=time.time(),
            supporting_patterns=["exploration", "novelty_seeking"]
        )
        
        self.assertEqual(trait.trait_name, "curious")
        self.assertEqual(trait.strength, 0.7)
        self.assertEqual(trait.confidence, 0.8)
        self.assertEqual(trait.evidence_count, 5)
        self.assertEqual(len(trait.supporting_patterns), 2)
    
    def test_identity_trait_serialization(self):
        """Test identity trait to_dict and from_dict."""
        trait = IdentityTrait(
            trait_name="analytical",
            strength=0.6,
            confidence=0.9,
            evidence_count=10,
            last_updated=time.time(),
            supporting_patterns=["strategy_selection", "optimization"]
        )
        
        data = trait.to_dict()
        self.assertIn('trait_name', data)
        self.assertIn('strength', data)
        self.assertIn('confidence', data)
        self.assertIn('supporting_patterns', data)
        
        restored_trait = IdentityTrait.from_dict(data)
        self.assertEqual(restored_trait.trait_name, trait.trait_name)
        self.assertEqual(restored_trait.strength, trait.strength)
        self.assertEqual(restored_trait.confidence, trait.confidence)
        self.assertEqual(restored_trait.supporting_patterns, trait.supporting_patterns)


class TestSelfAwareModel(unittest.TestCase):
    """Test cases for SelfAwareModel identity functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.test_db.close()
        self.db_path = self.test_db.name
        
        # Mock emotion model to avoid dependencies
        with patch('storage.self_model.get_emotion_model') as mock_emotion:
            mock_emotion.return_value = MagicMock()
            self.self_model = SelfAwareModel(self.db_path)
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_self_aware_model_initialization(self):
        """Test SelfAwareModel initialization."""
        self.assertIsInstance(self.self_model.identity_traits, dict)
        self.assertEqual(len(self.self_model.identity_traits), 0)
        self.assertGreater(self.self_model.identity_update_interval, 0)
        self.assertIsInstance(self.self_model.trait_patterns, dict)
    
    def test_trait_evidence_extraction(self):
        """Test extracting trait evidence from behavior."""
        # Test drive activation evidence
        evidence = self.self_model._extract_trait_evidence(
            "drive_activation",
            {"drive_name": "curiosity", "strength": 0.8}
        )
        
        # Should find evidence for curious trait
        self.assertIn("curious", evidence)
        self.assertGreater(evidence["curious"], 0)
        
        # Test strategy selection evidence
        evidence = self.self_model._extract_trait_evidence(
            "strategy_selection",
            {"strategy": "belief_clarification", "success_rate": 0.9}
        )
        
        # Should find evidence for analytical trait
        self.assertIn("analytical", evidence)
        self.assertGreater(evidence["analytical"], 0)
        
        # Test emotion event evidence
        evidence = self.self_model._extract_trait_evidence(
            "emotion_event",
            {"emotion_type": "curiosity", "strength": 0.7}
        )
        
        # Should find evidence for curious trait
        self.assertIn("curious", evidence)
        self.assertGreater(evidence["curious"], 0)
    
    def test_identity_trait_update(self):
        """Test updating identity traits from evidence."""
        # Add evidence for curious trait
        self.self_model._update_identity_trait("curious", 0.7, "drive_activation")
        
        # Should create new trait
        self.assertIn("curious", self.self_model.identity_traits)
        trait = self.self_model.identity_traits["curious"]
        self.assertEqual(trait.trait_name, "curious")
        self.assertGreater(trait.strength, 0)
        self.assertEqual(trait.evidence_count, 1)
        self.assertIn("drive_activation", trait.supporting_patterns)
        
        # Add more evidence for same trait
        old_strength = trait.strength
        self.self_model._update_identity_trait("curious", 0.8, "strategy_selection")
        
        # Should update existing trait
        updated_trait = self.self_model.identity_traits["curious"]
        self.assertEqual(updated_trait.evidence_count, 2)
        self.assertIn("strategy_selection", updated_trait.supporting_patterns)
        # Strength should be influenced by new evidence
        self.assertNotEqual(updated_trait.strength, old_strength)
    
    def test_trait_decay(self):
        """Test trait decay over time."""
        # Create a trait
        trait = IdentityTrait(
            trait_name="test_trait",
            strength=0.8,
            confidence=0.7,
            evidence_count=5,
            last_updated=time.time() - 86400 * 2,  # 2 days ago
            supporting_patterns=["test_pattern"]
        )
        self.self_model.identity_traits["test_trait"] = trait
        
        # Apply decay
        self.self_model._decay_unreinforced_traits()
        
        # Trait should be weakened or removed
        if "test_trait" in self.self_model.identity_traits:
            decayed_trait = self.self_model.identity_traits["test_trait"]
            self.assertLess(decayed_trait.strength, 0.8)
        # If trait was very weak, it might have been removed entirely
    
    def test_identity_signature_generation(self):
        """Test generating identity signature string."""
        # Initially should have no significant traits
        signature = self.self_model.get_identity_signature()
        self.assertIn("developing identity", signature.lower())
        
        # Add some strong traits
        strong_trait = IdentityTrait(
            trait_name="curious",
            strength=0.8,
            confidence=0.9,
            evidence_count=10,
            last_updated=time.time(),
            supporting_patterns=["exploration", "novelty"]
        )
        self.self_model.identity_traits["curious"] = strong_trait
        
        moderate_trait = IdentityTrait(
            trait_name="analytical",
            strength=0.6,
            confidence=0.8,
            evidence_count=8,
            last_updated=time.time(),
            supporting_patterns=["strategy", "optimization"]
        )
        self.self_model.identity_traits["analytical"] = moderate_trait
        
        # Generate signature
        signature = self.self_model.get_identity_signature()
        
        # Should contain trait names
        self.assertIn("curious", signature.lower())
        self.assertIn("analytical", signature.lower())
        
        # Should have descriptive qualifiers
        self.assertTrue(any(word in signature.lower() for word in ["very", "quite", "somewhat"]))
    
    @patch('storage.self_model.get_emotion_model')
    def test_emotional_influence_on_decisions(self, mock_get_emotion):
        """Test emotional influence calculation."""
        # Mock emotion model
        mock_emotion_model = MagicMock()
        mock_emotion_model.get_current_mood.return_value = {
            "mood_label": "curious",
            "valence": 0.6,
            "arousal": 0.5,
            "confidence": 0.8
        }
        mock_get_emotion.return_value = mock_emotion_model
        
        # Test emotional influence calculation
        influence = self.self_model.get_emotional_influence_on_decision({
            "decision_type": "strategy_selection"
        })
        
        self.assertIn("risk_tolerance", influence)
        self.assertIn("exploration_bias", influence)
        self.assertIn("patience", influence)
        self.assertIn("novelty_seeking", influence)
        
        # All values should be in reasonable range
        for key, value in influence.items():
            self.assertGreaterEqual(value, 0.1)
            self.assertLessEqual(value, 0.9)
    
    def test_update_identity_from_behavior(self):
        """Test complete behavior-to-identity update process."""
        # Test with recent update (should skip)
        self.self_model.last_identity_update = time.time() - 1800  # 30 minutes ago
        
        # This should be skipped due to recent update
        self.self_model.update_identity_from_behavior(
            "drive_activation",
            {"drive_name": "curiosity", "strength": 0.8}
        )
        
        # No traits should be added yet
        self.assertEqual(len(self.self_model.identity_traits), 0)
        
        # Force update by setting old timestamp
        self.self_model.last_identity_update = time.time() - 7200  # 2 hours ago
        
        # Now update should proceed
        self.self_model.update_identity_from_behavior(
            "drive_activation",
            {"drive_name": "curiosity", "strength": 0.8}
        )
        
        # Should have created curious trait
        self.assertIn("curious", self.self_model.identity_traits)
    
    def test_trait_confidence_evolution(self):
        """Test that trait confidence evolves with evidence."""
        # Create trait with low confidence
        trait = IdentityTrait(
            trait_name="resilient",
            strength=0.6,
            confidence=0.3,
            evidence_count=2,
            last_updated=time.time(),
            supporting_patterns=["stability"]
        )
        self.self_model.identity_traits["resilient"] = trait
        
        # Add more evidence
        for _ in range(10):
            self.self_model._update_identity_trait("resilient", 0.7, "pattern")
        
        # Confidence should have increased
        updated_trait = self.self_model.identity_traits["resilient"]
        self.assertGreater(updated_trait.confidence, 0.3)
        self.assertGreater(updated_trait.evidence_count, 2)
    
    def test_trait_pattern_mappings(self):
        """Test that trait patterns are correctly mapped."""
        # Verify all expected patterns exist
        expected_traits = ["curious", "analytical", "empathetic", "skeptical", "optimistic", "resilient"]
        
        for trait in expected_traits:
            self.assertIn(trait, self.self_model.trait_patterns)
            
            trait_config = self.self_model.trait_patterns[trait]
            self.assertIsInstance(trait_config, dict)
            
            # Should have at least one type of pattern
            has_patterns = any(key.endswith("_patterns") for key in trait_config.keys())
            self.assertTrue(has_patterns, f"Trait {trait} should have pattern mappings")
    
    def test_identity_database_operations(self):
        """Test database operations for identity traits."""
        # Add a trait
        trait = IdentityTrait(
            trait_name="test_trait",
            strength=0.7,
            confidence=0.8,
            evidence_count=5,
            last_updated=time.time(),
            supporting_patterns=["test_pattern"]
        )
        self.self_model.identity_traits["test_trait"] = trait
        
        # Save to database
        self.self_model._save_identity_traits()
        
        # Create new model instance and load
        with patch('storage.self_model.get_emotion_model') as mock_emotion:
            mock_emotion.return_value = MagicMock()
            new_model = SelfAwareModel(self.db_path)
        
        # Should have loaded the trait
        self.assertIn("test_trait", new_model.identity_traits)
        loaded_trait = new_model.identity_traits["test_trait"]
        self.assertEqual(loaded_trait.trait_name, "test_trait")
        self.assertEqual(loaded_trait.strength, 0.7)
        self.assertEqual(loaded_trait.confidence, 0.8)


class TestIdentityIntegration(unittest.TestCase):
    """Integration tests for identity signature system."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.test_db.close()
        self.db_path = self.test_db.name
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_get_self_aware_model_singleton(self):
        """Test that get_self_aware_model returns singleton."""
        with patch('storage.self_model.get_emotion_model') as mock_emotion:
            mock_emotion.return_value = MagicMock()
            
            model1 = get_self_aware_model(self.db_path)
            model2 = get_self_aware_model(self.db_path)
            
            # Should be the same instance
            self.assertIs(model1, model2)
    
    @patch('storage.self_model.get_emotion_model')
    def test_identity_evolution_simulation(self, mock_get_emotion):
        """Test simulated identity evolution over time."""
        # Mock emotion model
        mock_emotion_model = MagicMock()
        mock_emotion_model.get_current_mood.return_value = {
            "mood_label": "curious",
            "valence": 0.4,
            "arousal": 0.6
        }
        mock_get_emotion.return_value = mock_emotion_model
        
        model = SelfAwareModel(self.db_path)
        
        # Simulate repeated curious behavior
        model.last_identity_update = 0  # Force updates
        
        for i in range(10):
            model.update_identity_from_behavior(
                "drive_activation",
                {"drive_name": "curiosity", "strength": 0.7 + (i * 0.02)}
            )
            # Reset timestamp to force updates
            model.last_identity_update = 0
        
        # Should have developed curious trait
        self.assertIn("curious", model.identity_traits)
        curious_trait = model.identity_traits["curious"]
        self.assertGreater(curious_trait.strength, 0.5)
        self.assertGreater(curious_trait.confidence, 0.5)
        
        # Identity signature should reflect this
        signature = model.get_identity_signature()
        self.assertIn("curious", signature.lower())
    
    @patch('storage.self_model.get_emotion_model')
    def test_mixed_trait_development(self, mock_get_emotion):
        """Test development of multiple traits."""
        # Mock emotion model
        mock_emotion_model = MagicMock()
        mock_emotion_model.get_current_mood.return_value = {
            "mood_label": "analytical", 
            "valence": 0.2,
            "arousal": 0.4
        }
        mock_get_emotion.return_value = mock_emotion_model
        
        model = SelfAwareModel(self.db_path)
        model.last_identity_update = 0
        
        # Simulate mixed behaviors
        behaviors = [
            ("drive_activation", {"drive_name": "curiosity", "strength": 0.8}),
            ("strategy_selection", {"strategy": "belief_clarification", "success_rate": 0.9}),
            ("drive_activation", {"drive_name": "coherence", "strength": 0.7}),
            ("strategy_selection", {"strategy": "fact_consolidation", "success_rate": 0.8}),
            ("emotion_event", {"emotion_type": "satisfaction", "strength": 0.6})
        ]
        
        for behavior_type, behavior_data in behaviors:
            model.update_identity_from_behavior(behavior_type, behavior_data)
            model.last_identity_update = 0  # Force updates
        
        # Should have developed multiple traits
        self.assertGreater(len(model.identity_traits), 1)
        
        # Identity signature should be complex
        signature = model.get_identity_signature()
        self.assertGreater(len(signature), 20)  # Should be a substantial description


if __name__ == '__main__':
    unittest.main()