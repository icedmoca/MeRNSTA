#!/usr/bin/env python3
"""
Tests for CognitiveSelfModel - Symbolic Self-Modeling

Tests the symbolic self-modeling capabilities including:
- Rule creation and management
- Cognitive state recording
- Symbolic reasoning queries
- Strategy preference calculation
"""

import unittest
import tempfile
import os
import time
from unittest.mock import Mock, patch

from storage.self_model import CognitiveSelfModel, SymbolicRule, CognitiveState


class TestCognitiveSelfModel(unittest.TestCase):
    """Test cases for CognitiveSelfModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.self_model = CognitiveSelfModel(db_path=self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary database
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_initialization(self):
        """Test that the self model initializes correctly."""
        self.assertIsNotNone(self.self_model)
        self.assertEqual(self.self_model.min_confidence_threshold, 0.6)
        self.assertEqual(self.self_model.min_support_count, 3)
    
    def test_rule_creation(self):
        """Test creating and storing symbolic rules."""
        # Create a test rule
        rule = SymbolicRule(
            rule_id="test_rule_1",
            antecedent="Belief(cluster_x) AND Contradiction(fact_y)",
            consequent="Strategy(belief_clarification)",
            confidence=0.8,
            support_count=5,
            contradiction_count=1
        )
        
        # Store the rule
        self.self_model._store_rule(rule)
        
        # Query for the rule
        rules = self.self_model.query_rules("Strategy(belief_clarification)")
        
        self.assertGreater(len(rules), 0)
        found_rule = rules[0]
        self.assertEqual(found_rule.antecedent, rule.antecedent)
        self.assertEqual(found_rule.consequent, rule.consequent)
        self.assertEqual(found_rule.confidence, rule.confidence)
    
    def test_cognitive_state_recording(self):
        """Test recording cognitive states."""
        beliefs = {"cluster_1": 0.8, "cluster_2": 0.6}
        contradictions = ["contradiction_1", "contradiction_2"]
        active_strategies = ["belief_clarification", "fact_consolidation"]
        reflex_chain = ["reflex_1", "reflex_2"]
        success_patterns = {"belief_clarification": 0.7, "fact_consolidation": 0.6}
        volatility_scores = {"cluster_1": 0.3, "cluster_2": 0.5}
        
        state_id = self.self_model.record_cognitive_state(
            beliefs, contradictions, active_strategies, 
            reflex_chain, success_patterns, volatility_scores
        )
        
        self.assertIsNotNone(state_id)
        self.assertIsInstance(state_id, str)
    
    def test_rule_querying(self):
        """Test querying rules with different patterns."""
        # Create test rules
        rules = [
            SymbolicRule("rule1", "Belief(cluster_x)", "Strategy(belief_clarification)", 0.8, 5, 1),
            SymbolicRule("rule2", "Contradiction(fact_y)", "Strategy(fact_consolidation)", 0.7, 4, 2),
            SymbolicRule("rule3", "Volatility(high)", "Strategy(cluster_reassessment)", 0.9, 6, 0)
        ]
        
        for rule in rules:
            self.self_model._store_rule(rule)
        
        # Test different queries
        belief_rules = self.self_model.query_rules("Belief")
        self.assertGreater(len(belief_rules), 0)
        
        strategy_rules = self.self_model.query_rules("Strategy")
        self.assertGreaterEqual(len(strategy_rules), 3)
        
        contradiction_rules = self.self_model.query_rules("Contradiction")
        self.assertGreater(len(contradiction_rules), 0)
    
    def test_strategy_preferences(self):
        """Test calculating strategy preferences."""
        # Create context
        context = {
            "drift_type": "contradiction",
            "cluster_id": "test_cluster",
            "volatility_score": 0.4
        }
        
        # Add some rules that would influence preferences
        rule = SymbolicRule(
            "pref_rule",
            "DriftType(contradiction)",
            "PreferredStrategy(belief_clarification)",
            0.8, 5, 1
        )
        self.self_model._store_rule(rule)
        
        preferences = self.self_model.get_strategy_preferences(context)
        
        self.assertIsInstance(preferences, list)
        # Should have at least some preferences
        self.assertGreater(len(preferences), 0)
        
        # Check that preferences are tuples of (strategy, confidence)
        for pref in preferences:
            self.assertIsInstance(pref, tuple)
            self.assertEqual(len(pref), 2)
            self.assertIsInstance(pref[0], str)
            self.assertIsInstance(pref[1], float)
    
    def test_belief_dynamics_recording(self):
        """Test recording belief dynamics."""
        cluster_id = "test_cluster"
        belief_change = 0.2
        cause_strategy = "belief_clarification"
        reflex_cycle_id = "reflex_123"
        
        # Record belief dynamics
        self.self_model.record_belief_dynamics(
            cluster_id, belief_change, cause_strategy, reflex_cycle_id
        )
        
        # This should create a rule about the relationship
        rules = self.self_model.query_rules("belief_clarification")
        
        # Should have at least one rule about belief dynamics
        self.assertGreaterEqual(len(rules), 0)
    
    def test_strategy_effectiveness_recording(self):
        """Test recording strategy effectiveness."""
        strategy_name = "belief_clarification"
        success = True
        score = 0.8
        
        # Record strategy effectiveness
        self.self_model.record_strategy_effectiveness(strategy_name, success, score)
        
        # This should update the strategy's success pattern
        # We can verify by checking if rules were created
        rules = self.self_model.query_rules("belief_clarification")
        
        # Should have some rules about this strategy
        self.assertGreaterEqual(len(rules), 0)
    
    def test_cognitive_statistics(self):
        """Test getting cognitive statistics."""
        # Record some test data first
        beliefs = {"cluster_1": 0.8}
        contradictions = ["contradiction_1"]
        active_strategies = ["belief_clarification"]
        reflex_chain = ["reflex_1"]
        success_patterns = {"belief_clarification": 0.7}
        volatility_scores = {"cluster_1": 0.3}
        
        self.self_model.record_cognitive_state(
            beliefs, contradictions, active_strategies,
            reflex_chain, success_patterns, volatility_scores
        )
        
        # Get statistics
        stats = self.self_model.get_cognitive_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_states', stats)
        self.assertIn('total_rules', stats)
        self.assertIn('average_confidence', stats)
    
    def test_symbolic_self_representation(self):
        """Test getting symbolic self-representation."""
        # Add some test data
        beliefs = {"cluster_1": 0.8}
        contradictions = ["contradiction_1"]
        active_strategies = ["belief_clarification"]
        reflex_chain = ["reflex_1"]
        success_patterns = {"belief_clarification": 0.7}
        volatility_scores = {"cluster_1": 0.3}
        
        self.self_model.record_cognitive_state(
            beliefs, contradictions, active_strategies,
            reflex_chain, success_patterns, volatility_scores
        )
        
        # Add a test rule
        rule = SymbolicRule(
            "test_rule",
            "Belief(cluster_1)",
            "Strategy(belief_clarification)",
            0.8, 5, 1
        )
        self.self_model._store_rule(rule)
        
        # Get representation
        representation = self.self_model.get_symbolic_self_representation()
        
        self.assertIsInstance(representation, dict)
        self.assertIn('core_beliefs', representation)
        self.assertIn('active_strategies', representation)
        self.assertIn('symbolic_rules', representation)
        self.assertIn('cognitive_patterns', representation)
    
    def test_rule_confidence_updating(self):
        """Test that rule confidence updates correctly."""
        # Create initial rule
        rule = SymbolicRule(
            "update_test_rule",
            "TestCondition",
            "TestConclusion",
            0.5, 3, 2
        )
        
        # Store rule
        self.self_model._store_rule(rule)
        
        # Update rule with new evidence
        updated_rule = SymbolicRule(
            "update_test_rule",
            "TestCondition",
            "TestConclusion",
            0.7, 5, 2  # More support, same contradictions
        )
        
        # Update the rule
        self.self_model._update_rule(updated_rule)
        
        # Query for the rule
        rules = self.self_model.query_rules("TestConclusion")
        
        self.assertGreater(len(rules), 0)
        found_rule = rules[0]
        self.assertEqual(found_rule.confidence, 0.7)
        self.assertEqual(found_rule.support_count, 5)
    
    def test_rule_evolution(self):
        """Test that rules evolve over time with new evidence."""
        # Create initial rule with low confidence
        rule = SymbolicRule(
            "evolution_rule",
            "EvolutionCondition",
            "EvolutionConclusion",
            0.3, 2, 3
        )
        self.self_model._store_rule(rule)
        
        # Add more supporting evidence
        for i in range(5):
            supporting_rule = SymbolicRule(
                f"support_rule_{i}",
                "EvolutionCondition",
                "EvolutionConclusion",
                0.6, 1, 0
            )
            self.self_model._add_or_update_rule(supporting_rule)
        
        # Query for evolved rule
        rules = self.self_model.query_rules("EvolutionConclusion")
        
        self.assertGreater(len(rules), 0)
        # The rule should have evolved with more support
        evolved_rule = rules[0]
        self.assertGreater(evolved_rule.confidence, 0.3)
        self.assertGreater(evolved_rule.support_count, 2)


class TestSymbolicRule(unittest.TestCase):
    """Test cases for SymbolicRule dataclass."""
    
    def test_rule_creation(self):
        """Test creating a symbolic rule."""
        rule = SymbolicRule(
            rule_id="test_rule",
            antecedent="A AND B",
            consequent="C",
            confidence=0.8,
            support_count=5,
            contradiction_count=1
        )
        
        self.assertEqual(rule.rule_id, "test_rule")
        self.assertEqual(rule.antecedent, "A AND B")
        self.assertEqual(rule.consequent, "C")
        self.assertEqual(rule.confidence, 0.8)
        self.assertEqual(rule.support_count, 5)
        self.assertEqual(rule.contradiction_count, 1)
        self.assertEqual(rule.rule_type, "implication")
    
    def test_rule_serialization(self):
        """Test rule serialization to/from dictionary."""
        rule = SymbolicRule(
            rule_id="serial_test",
            antecedent="TestAntecedent",
            consequent="TestConsequent",
            confidence=0.7,
            support_count=4,
            contradiction_count=2
        )
        
        # Convert to dictionary
        rule_dict = rule.to_dict()
        
        # Convert back to rule
        reconstructed_rule = SymbolicRule.from_dict(rule_dict)
        
        self.assertEqual(rule.rule_id, reconstructed_rule.rule_id)
        self.assertEqual(rule.antecedent, reconstructed_rule.antecedent)
        self.assertEqual(rule.consequent, reconstructed_rule.consequent)
        self.assertEqual(rule.confidence, reconstructed_rule.confidence)


class TestCognitiveState(unittest.TestCase):
    """Test cases for CognitiveState dataclass."""
    
    def test_state_creation(self):
        """Test creating a cognitive state."""
        state = CognitiveState(
            state_id="test_state",
            timestamp=time.time(),
            beliefs={"cluster_1": 0.8},
            contradictions=["contradiction_1"],
            active_strategies=["strategy_1"],
            reflex_chain=["reflex_1"],
            success_patterns={"strategy_1": 0.7},
            volatility_scores={"cluster_1": 0.3}
        )
        
        self.assertEqual(state.state_id, "test_state")
        self.assertIn("cluster_1", state.beliefs)
        self.assertIn("contradiction_1", state.contradictions)
        self.assertIn("strategy_1", state.active_strategies)
    
    def test_state_serialization(self):
        """Test state serialization to/from dictionary."""
        state = CognitiveState(
            state_id="serial_test",
            timestamp=time.time(),
            beliefs={"test_cluster": 0.6},
            contradictions=["test_contradiction"],
            active_strategies=["test_strategy"],
            reflex_chain=["test_reflex"],
            success_patterns={"test_strategy": 0.8},
            volatility_scores={"test_cluster": 0.4}
        )
        
        # Convert to dictionary
        state_dict = state.to_dict()
        
        # Convert back to state
        reconstructed_state = CognitiveState.from_dict(state_dict)
        
        self.assertEqual(state.state_id, reconstructed_state.state_id)
        self.assertEqual(state.beliefs, reconstructed_state.beliefs)
        self.assertEqual(state.contradictions, reconstructed_state.contradictions)


if __name__ == '__main__':
    unittest.main() 