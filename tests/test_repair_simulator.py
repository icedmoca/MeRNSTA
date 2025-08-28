#!/usr/bin/env python3
"""
Tests for RepairSimulator - Symbolic Repair Path Simulation

Tests the repair simulation capabilities including:
- Strategy simulation
- Outcome prediction
- Risk factor identification
- Confidence calculation
"""

import unittest
import tempfile
import os
import time
from unittest.mock import Mock, patch, MagicMock

from agents.repair_simulator import (
    RepairSimulator, SimulatedRepair, RepairSimulationResult,
    get_repair_simulator, simulate_repair_paths
)
from storage.reflex_log import ReflexScore


class TestRepairSimulator(unittest.TestCase):
    """Test cases for RepairSimulator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock reflex logger and self model
        self.mock_reflex_logger = Mock()
        self.mock_self_model = Mock()
        
        self.simulator = RepairSimulator(
            reflex_logger=self.mock_reflex_logger,
            self_model=self.mock_self_model
        )
    
    def test_initialization(self):
        """Test that the repair simulator initializes correctly."""
        self.assertIsNotNone(self.simulator)
        self.assertEqual(self.simulator.min_confidence_threshold, 0.3)
        self.assertEqual(len(self.simulator.available_strategies), 4)
        self.assertIn('belief_clarification', self.simulator.available_strategies)
        self.assertIn('cluster_reassessment', self.simulator.available_strategies)
        self.assertIn('fact_consolidation', self.simulator.available_strategies)
        self.assertIn('anticipatory_drift', self.simulator.available_strategies)
    
    def test_simulate_repair_paths(self):
        """Test simulating multiple repair paths."""
        # Mock historical scores
        mock_scores = [
            ReflexScore("cycle1", True, "belief_clarification", 123, 
                       coherence_delta=0.1, volatility_delta=-0.05, 
                       belief_consistency_delta=0.2, score=0.8),
            ReflexScore("cycle2", True, "belief_clarification", 123,
                       coherence_delta=0.15, volatility_delta=-0.1,
                       belief_consistency_delta=0.25, score=0.85)
        ]
        
        self.mock_reflex_logger.get_scores_by_strategy.return_value = mock_scores
        
        # Mock symbolic rules
        mock_rules = [Mock(confidence=0.8, consequent="Strategy(belief_clarification)")]
        self.mock_self_model.query_rules.return_value = mock_rules
        
        # Test simulation
        goal_id = "test_goal"
        current_state = {
            'drift_type': 'contradiction',
            'drift_score': 0.6,
            'cluster_id': 'test_cluster',
            'volatility_score': 0.4,
            'contradiction_count': 3,
            'coherence_score': 0.7,
            'token_id': 123
        }
        
        result = self.simulator.simulate_repair_paths(goal_id, current_state)
        
        self.assertIsInstance(result, RepairSimulationResult)
        self.assertEqual(result.goal_id, goal_id)
        self.assertEqual(result.current_state, current_state)
        self.assertGreater(len(result.simulated_repairs), 0)
        self.assertIsNotNone(result.best_repair)
        self.assertIsInstance(result.reasoning_summary, str)
    
    def test_simulate_single_strategy(self):
        """Test simulating a single repair strategy."""
        # Mock historical scores
        mock_scores = [
            ReflexScore("cycle1", True, "belief_clarification", 123,
                       coherence_delta=0.1, volatility_delta=-0.05,
                       belief_consistency_delta=0.2, score=0.8)
        ]
        
        self.mock_reflex_logger.get_scores_by_strategy.return_value = mock_scores
        
        # Mock symbolic rules
        mock_rules = [Mock(confidence=0.8, consequent="Strategy(belief_clarification)")]
        self.mock_self_model.query_rules.return_value = mock_rules
        
        # Test single strategy simulation
        strategy = "belief_clarification"
        goal_id = "test_goal"
        current_state = {
            'drift_type': 'contradiction',
            'drift_score': 0.6,
            'cluster_id': 'test_cluster'
        }
        
        result = self.simulator._simulate_single_strategy(strategy, goal_id, current_state)
        
        self.assertIsInstance(result, SimulatedRepair)
        self.assertEqual(result.strategy, strategy)
        self.assertGreater(result.predicted_score, 0)
        self.assertGreater(result.confidence, 0)
        self.assertIsInstance(result.reasoning_chain, list)
        self.assertIsInstance(result.risk_factors, list)
        self.assertGreater(result.success_probability, 0)
    
    def test_calculate_base_predictions(self):
        """Test calculating base predictions from historical scores."""
        # Create test historical scores
        historical_scores = [
            ReflexScore("cycle1", True, "belief_clarification", 123,
                       coherence_delta=0.1, volatility_delta=-0.05,
                       belief_consistency_delta=0.2, score=0.8),
            ReflexScore("cycle2", True, "belief_clarification", 123,
                       coherence_delta=0.15, volatility_delta=-0.1,
                       belief_consistency_delta=0.25, score=0.85)
        ]
        
        predictions = self.simulator._calculate_base_predictions(historical_scores)
        
        self.assertIn('score', predictions)
        self.assertIn('coherence_delta', predictions)
        self.assertIn('volatility_delta', predictions)
        self.assertIn('consistency_delta', predictions)
        
        # Check that predictions are reasonable
        self.assertGreater(predictions['score'], 0)
        self.assertLess(predictions['score'], 1)
    
    def test_apply_symbolic_reasoning(self):
        """Test applying symbolic reasoning adjustments."""
        # Mock symbolic rules
        mock_rules = [
            Mock(confidence=0.8, consequent="Strategy(belief_clarification)"),
            Mock(confidence=0.7, consequent="CoherenceImprovement")
        ]
        self.mock_self_model.query_rules.return_value = mock_rules
        
        strategy = "belief_clarification"
        current_state = {
            'drift_type': 'contradiction',
            'drift_score': 0.6
        }
        
        adjustments = self.simulator._apply_symbolic_reasoning(strategy, current_state)
        
        self.assertIn('score', adjustments)
        self.assertIn('coherence_delta', adjustments)
        self.assertIn('volatility_delta', adjustments)
        self.assertIn('consistency_delta', adjustments)
        
        # Check that adjustments are applied
        self.assertGreater(adjustments['consistency_delta'], 0)
        self.assertGreater(adjustments['score'], 0)
    
    def test_apply_context_similarity(self):
        """Test applying context similarity adjustments."""
        # Create test historical scores with cluster information
        historical_scores = [
            ReflexScore("cycle1", True, "belief_clarification", 123,
                       coherence_delta=0.1, volatility_delta=-0.05,
                       belief_consistency_delta=0.2, score=0.8,
                       cluster_id="test_cluster"),
            ReflexScore("cycle2", True, "belief_clarification", 123,
                       coherence_delta=0.15, volatility_delta=-0.1,
                       belief_consistency_delta=0.25, score=0.85,
                       cluster_id="different_cluster")
        ]
        
        strategy = "belief_clarification"
        current_state = {
            'cluster_id': 'test_cluster',
            'drift_score': 0.6,
            'token_id': 123
        }
        
        adjustments = self.simulator._apply_context_similarity(strategy, current_state, historical_scores)
        
        self.assertIn('score', adjustments)
        self.assertIn('coherence_delta', adjustments)
        self.assertIn('volatility_delta', adjustments)
        self.assertIn('consistency_delta', adjustments)
    
    def test_identify_risk_factors(self):
        """Test identifying risk factors for strategies."""
        # Create test historical scores
        historical_scores = [
            ReflexScore("cycle1", False, "belief_clarification", 123, score=0.3),
            ReflexScore("cycle2", True, "belief_clarification", 123, score=0.8)
        ]
        
        strategy = "belief_clarification"
        current_state = {
            'volatility_score': 0.8,  # High volatility
            'contradiction_count': 8,  # High contradiction count
            'belief_confidence': 0.2   # Low belief confidence
        }
        
        risk_factors = self.simulator._identify_risk_factors(strategy, current_state, historical_scores)
        
        self.assertIsInstance(risk_factors, list)
        self.assertGreater(len(risk_factors), 0)
        
        # Should identify high volatility risk
        volatility_risks = [r for r in risk_factors if 'volatility' in r.lower()]
        self.assertGreater(len(volatility_risks), 0)
        
        # Should identify high contradiction risk
        contradiction_risks = [r for r in risk_factors if 'contradiction' in r.lower()]
        self.assertGreater(len(contradiction_risks), 0)
    
    def test_combine_predictions(self):
        """Test combining different prediction components."""
        base_predictions = {
            'score': 0.7,
            'coherence_delta': 0.1,
            'volatility_delta': -0.05,
            'consistency_delta': 0.2
        }
        
        symbolic_adjustments = {
            'score': 0.1,
            'coherence_delta': 0.05,
            'volatility_delta': -0.02,
            'consistency_delta': 0.1
        }
        
        context_adjustments = {
            'score': 0.05,
            'coherence_delta': 0.02,
            'volatility_delta': -0.01,
            'consistency_delta': 0.05
        }
        
        combined = self.simulator._combine_predictions(
            base_predictions, symbolic_adjustments, context_adjustments
        )
        
        self.assertIn('score', combined)
        self.assertIn('coherence_delta', combined)
        self.assertIn('volatility_delta', combined)
        self.assertIn('consistency_delta', combined)
        
        # Check that predictions are within bounds
        self.assertGreaterEqual(combined['score'], 0.0)
        self.assertLessEqual(combined['score'], 1.0)
        self.assertGreaterEqual(combined['coherence_delta'], -1.0)
        self.assertLessEqual(combined['coherence_delta'], 1.0)
    
    def test_calculate_prediction_confidence(self):
        """Test calculating prediction confidence."""
        # Create test historical scores
        historical_scores = [
            ReflexScore("cycle1", True, "belief_clarification", 123, score=0.8),
            ReflexScore("cycle2", True, "belief_clarification", 123, score=0.85),
            ReflexScore("cycle3", True, "belief_clarification", 123, score=0.9)
        ]
        
        symbolic_adjustments = {'score': 0.1, 'coherence_delta': 0.05, 'volatility_delta': -0.02, 'consistency_delta': 0.1}
        context_adjustments = {'score': 0.05, 'coherence_delta': 0.02, 'volatility_delta': -0.01, 'consistency_delta': 0.05}
        
        confidence = self.simulator._calculate_prediction_confidence(
            historical_scores, symbolic_adjustments, context_adjustments
        )
        
        self.assertGreater(confidence, 0)
        self.assertLessEqual(confidence, 1)
    
    def test_calculate_success_probability(self):
        """Test calculating success probability."""
        # Create test historical scores with mixed success
        historical_scores = [
            ReflexScore("cycle1", True, "belief_clarification", 123, score=0.8),
            ReflexScore("cycle2", False, "belief_clarification", 123, score=0.3),
            ReflexScore("cycle3", True, "belief_clarification", 123, score=0.9)
        ]
        
        risk_factors = ["High volatility", "Low belief confidence"]
        confidence = 0.7
        
        probability = self.simulator._calculate_success_probability(historical_scores, risk_factors, confidence)
        
        self.assertGreater(probability, 0)
        self.assertLessEqual(probability, 1)
    
    def test_estimate_duration(self):
        """Test estimating execution duration."""
        # Test with historical data
        historical_scores = [
            ReflexScore("cycle1", True, "belief_clarification", 123, score=0.8),
            ReflexScore("cycle2", True, "belief_clarification", 123, score=0.85)
        ]
        
        duration = self.simulator._estimate_duration("belief_clarification", historical_scores)
        
        self.assertGreater(duration, 0)
        
        # Test with no historical data (should use default)
        duration_default = self.simulator._estimate_duration("belief_clarification", [])
        self.assertEqual(duration_default, 30.0)
    
    def test_create_default_simulation(self):
        """Test creating default simulation when no historical data is available."""
        strategy = "belief_clarification"
        current_state = {'drift_type': 'contradiction'}
        
        result = self.simulator._create_default_simulation(strategy, current_state)
        
        self.assertIsInstance(result, SimulatedRepair)
        self.assertEqual(result.strategy, strategy)
        self.assertEqual(result.predicted_score, 0.5)
        self.assertEqual(result.confidence, 0.3)
        self.assertIn("No historical data available", result.reasoning_chain)
        self.assertIn("No historical performance data", result.risk_factors)
    
    def test_generate_reasoning_chain(self):
        """Test generating reasoning chain for simulation."""
        strategy = "belief_clarification"
        base_predictions = {'score': 0.7, 'coherence_delta': 0.1, 'volatility_delta': -0.05, 'consistency_delta': 0.2}
        symbolic_adjustments = {'score': 0.1, 'coherence_delta': 0.05, 'volatility_delta': -0.02, 'consistency_delta': 0.1}
        context_adjustments = {'score': 0.05, 'coherence_delta': 0.02, 'volatility_delta': -0.01, 'consistency_delta': 0.05}
        risk_factors = ["High volatility", "Low belief confidence"]
        
        chain = self.simulator._generate_reasoning_chain(
            strategy, base_predictions, symbolic_adjustments, context_adjustments, risk_factors
        )
        
        self.assertIsInstance(chain, list)
        self.assertGreater(len(chain), 0)
        self.assertIn(strategy, chain[0])
        self.assertIn("risk factors", chain[-1])
    
    def test_generate_reasoning_summary(self):
        """Test generating reasoning summary."""
        # Create test simulated repairs
        repairs = [
            SimulatedRepair("belief_clarification", 0.8, 0.1, -0.05, 0.2, 0.9, 
                           ["Strategy: belief_clarification"], 30.0, [], 0.85),
            SimulatedRepair("fact_consolidation", 0.6, 0.05, -0.02, 0.1, 0.7,
                           ["Strategy: fact_consolidation"], 45.0, ["Risk factor"], 0.7)
        ]
        
        best_repair = repairs[0]
        
        summary = self.simulator._generate_reasoning_summary(repairs, best_repair)
        
        self.assertIsInstance(summary, str)
        self.assertIn("2 repair strategies", summary)
        self.assertIn("belief_clarification", summary)
        self.assertIn("fact_consolidation", summary)
        self.assertIn("Best strategy", summary)


class TestSimulatedRepair(unittest.TestCase):
    """Test cases for SimulatedRepair dataclass."""
    
    def test_repair_creation(self):
        """Test creating a simulated repair."""
        repair = SimulatedRepair(
            strategy="belief_clarification",
            predicted_score=0.8,
            predicted_coherence_delta=0.1,
            predicted_volatility_delta=-0.05,
            predicted_consistency_delta=0.2,
            confidence=0.9,
            reasoning_chain=["Strategy: belief_clarification"],
            estimated_duration=30.0,
            risk_factors=["High volatility"],
            success_probability=0.85
        )
        
        self.assertEqual(repair.strategy, "belief_clarification")
        self.assertEqual(repair.predicted_score, 0.8)
        self.assertEqual(repair.confidence, 0.9)
        self.assertEqual(repair.estimated_duration, 30.0)
        self.assertEqual(repair.success_probability, 0.85)
        self.assertIn("High volatility", repair.risk_factors)
    
    def test_repair_serialization(self):
        """Test repair serialization to/from dictionary."""
        repair = SimulatedRepair(
            strategy="fact_consolidation",
            predicted_score=0.7,
            predicted_coherence_delta=0.05,
            predicted_volatility_delta=-0.02,
            predicted_consistency_delta=0.1,
            confidence=0.8,
            reasoning_chain=["Strategy: fact_consolidation"],
            estimated_duration=45.0,
            risk_factors=["Risk factor"],
            success_probability=0.75
        )
        
        # Convert to dictionary
        repair_dict = repair.to_dict()
        
        # Convert back to repair
        reconstructed_repair = SimulatedRepair.from_dict(repair_dict)
        
        self.assertEqual(repair.strategy, reconstructed_repair.strategy)
        self.assertEqual(repair.predicted_score, reconstructed_repair.predicted_score)
        self.assertEqual(repair.confidence, reconstructed_repair.confidence)
        self.assertEqual(repair.estimated_duration, reconstructed_repair.estimated_duration)


class TestRepairSimulationResult(unittest.TestCase):
    """Test cases for RepairSimulationResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a repair simulation result."""
        repairs = [
            SimulatedRepair("belief_clarification", 0.8, 0.1, -0.05, 0.2, 0.9,
                           ["Strategy: belief_clarification"], 30.0, [], 0.85)
        ]
        
        result = RepairSimulationResult(
            goal_id="test_goal",
            current_state={'drift_type': 'contradiction'},
            simulated_repairs=repairs,
            best_repair=repairs[0],
            reasoning_summary="Test reasoning",
            simulation_timestamp=time.time()
        )
        
        self.assertEqual(result.goal_id, "test_goal")
        self.assertEqual(len(result.simulated_repairs), 1)
        self.assertEqual(result.best_repair, repairs[0])
        self.assertEqual(result.reasoning_summary, "Test reasoning")
    
    def test_result_serialization(self):
        """Test result serialization to/from dictionary."""
        repairs = [
            SimulatedRepair("belief_clarification", 0.8, 0.1, -0.05, 0.2, 0.9,
                           ["Strategy: belief_clarification"], 30.0, [], 0.85)
        ]
        
        result = RepairSimulationResult(
            goal_id="test_goal",
            current_state={'drift_type': 'contradiction'},
            simulated_repairs=repairs,
            best_repair=repairs[0],
            reasoning_summary="Test reasoning",
            simulation_timestamp=time.time()
        )
        
        # Convert to dictionary
        result_dict = result.to_dict()
        
        # Convert back to result
        reconstructed_result = RepairSimulationResult.from_dict(result_dict)
        
        self.assertEqual(result.goal_id, reconstructed_result.goal_id)
        self.assertEqual(len(result.simulated_repairs), len(reconstructed_result.simulated_repairs))
        self.assertEqual(result.reasoning_summary, reconstructed_result.reasoning_summary)


class TestGlobalFunctions(unittest.TestCase):
    """Test cases for global functions."""
    
    @patch('agents.repair_simulator._repair_simulator_instance', None)
    def test_get_repair_simulator(self):
        """Test getting the global repair simulator instance."""
        simulator = get_repair_simulator()
        self.assertIsInstance(simulator, RepairSimulator)
    
    @patch('agents.repair_simulator.get_repair_simulator')
    def test_simulate_repair_paths(self):
        """Test the convenience function for simulating repair paths."""
        mock_simulator = Mock()
        mock_result = Mock()
        mock_simulator.simulate_repair_paths.return_value = mock_result
        
        with patch('agents.repair_simulator.get_repair_simulator', return_value=mock_simulator):
            result = simulate_repair_paths("test_goal", {'drift_type': 'contradiction'})
            
            self.assertEqual(result, mock_result)
            mock_simulator.simulate_repair_paths.assert_called_once()


if __name__ == '__main__':
    unittest.main() 