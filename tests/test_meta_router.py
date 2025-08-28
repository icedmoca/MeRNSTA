#!/usr/bin/env python3
"""
Tests for MetaRouter - Introspective Subgoal Routing

Tests the meta routing capabilities including:
- Subgoal generation from simulation results
- Agent routing decisions
- Dependency tracking
- Routing optimization
"""

import unittest
import tempfile
import os
import time
from unittest.mock import Mock, patch, MagicMock

from agents.meta_router import (
    MetaRouter, RoutedSubgoal, RoutingPlan,
    get_meta_router, route_subgoals
)
from agents.repair_simulator import SimulatedRepair, RepairSimulationResult


class TestMetaRouter(unittest.TestCase):
    """Test cases for MetaRouter."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock components
        self.mock_reflex_logger = Mock()
        self.mock_self_model = Mock()
        self.mock_repair_simulator = Mock()
        
        self.meta_router = MetaRouter(
            reflex_logger=self.mock_reflex_logger,
            self_model=self.mock_self_model,
            repair_simulator=self.mock_repair_simulator
        )
    
    def test_initialization(self):
        """Test that the meta router initializes correctly."""
        self.assertIsNotNone(self.meta_router)
        self.assertEqual(self.meta_router.min_confidence_threshold, 0.4)
        self.assertEqual(self.meta_router.max_subgoals_per_goal, 5)
        
        # Check agent capabilities
        self.assertIn('reflector', self.meta_router.agent_capabilities)
        self.assertIn('clarifier', self.meta_router.agent_capabilities)
        self.assertIn('consolidator', self.meta_router.agent_capabilities)
        self.assertIn('anticipator', self.meta_router.agent_capabilities)
        self.assertIn('optimizer', self.meta_router.agent_capabilities)
    
    def test_route_subgoals(self):
        """Test routing subgoals for a goal."""
        # Mock simulation result
        mock_repairs = [
            SimulatedRepair("belief_clarification", 0.8, 0.1, -0.05, 0.2, 0.9,
                           ["Strategy: belief_clarification"], 30.0, [], 0.85),
            SimulatedRepair("fact_consolidation", 0.6, 0.05, -0.02, 0.1, 0.7,
                           ["Strategy: fact_consolidation"], 45.0, ["Risk factor"], 0.7)
        ]
        
        mock_simulation_result = RepairSimulationResult(
            goal_id="test_goal",
            current_state={'drift_type': 'contradiction'},
            simulated_repairs=mock_repairs,
            best_repair=mock_repairs[0],
            reasoning_summary="Test simulation",
            simulation_timestamp=time.time()
        )
        
        self.mock_repair_simulator.simulate_repair_paths.return_value = mock_simulation_result
        
        # Mock symbolic rules
        mock_rules = [Mock(confidence=0.8, antecedent="DriftType(contradiction)")]
        self.mock_self_model.query_rules.return_value = mock_rules
        
        # Test routing
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
        
        result = self.meta_router.route_subgoals(
            goal_id, current_state, 
            goal_description="Test goal",
            tags=["test", "contradiction"]
        )
        
        self.assertIsInstance(result, RoutingPlan)
        self.assertEqual(result.goal_id, goal_id)
        self.assertGreater(len(result.subgoals), 0)
        self.assertIsInstance(result.total_estimated_duration, float)
        self.assertIsInstance(result.routing_confidence, float)
        self.assertIsInstance(result.reasoning_summary, str)
    
    def test_generate_subgoals_from_simulation(self):
        """Test generating subgoals from simulation results."""
        # Create mock simulation result
        mock_repairs = [
            SimulatedRepair("belief_clarification", 0.8, 0.1, -0.05, 0.2, 0.9,
                           ["Strategy: belief_clarification"], 30.0, [], 0.85),
            SimulatedRepair("fact_consolidation", 0.2, 0.05, -0.02, 0.1, 0.7,
                           ["Strategy: fact_consolidation"], 45.0, ["Risk factor"], 0.7)
        ]
        
        mock_simulation_result = RepairSimulationResult(
            goal_id="test_goal",
            current_state={'drift_type': 'contradiction'},
            simulated_repairs=mock_repairs,
            best_repair=mock_repairs[0],
            reasoning_summary="Test simulation",
            simulation_timestamp=time.time()
        )
        
        goal_id = "test_goal"
        current_state = {
            'drift_type': 'contradiction',
            'drift_score': 0.6,
            'cluster_id': 'test_cluster'
        }
        tags = ["test", "contradiction"]
        
        subgoals = self.meta_router._generate_subgoals_from_simulation(
            goal_id, mock_simulation_result, current_state, tags
        )
        
        self.assertIsInstance(subgoals, list)
        # Should only include strategies with score > 0.3
        self.assertGreater(len(subgoals), 0)
        self.assertLess(len(subgoals), len(mock_repairs))  # One strategy was filtered out
        
        for subgoal in subgoals:
            self.assertIn('subgoal_id', subgoal)
            self.assertIn('goal_id', subgoal)
            self.assertIn('strategy', subgoal)
            self.assertIn('predicted_score', subgoal)
            self.assertIn('confidence', subgoal)
            self.assertIn('priority', subgoal)
            self.assertIn('tags', subgoal)
    
    def test_generate_specialized_subgoals(self):
        """Test generating specialized subgoals based on current state."""
        goal_id = "test_goal"
        tags = ["test"]
        
        # Test with high contradiction count
        current_state_high_contradiction = {
            'contradiction_count': 5,  # High contradiction count
            'volatility_score': 0.3,
            'coherence_score': 0.8
        }
        
        specialized_subgoals = self.meta_router._generate_specialized_subgoals(
            goal_id, current_state_high_contradiction, tags
        )
        
        self.assertIsInstance(specialized_subgoals, list)
        self.assertGreater(len(specialized_subgoals), 0)
        
        # Should have contradiction resolution subgoal
        contradiction_subgoals = [s for s in specialized_subgoals if 'contradiction_resolution' in s['subgoal_id']]
        self.assertGreater(len(contradiction_subgoals), 0)
        
        # Test with high volatility
        current_state_high_volatility = {
            'contradiction_count': 1,
            'volatility_score': 0.8,  # High volatility
            'coherence_score': 0.8
        }
        
        specialized_subgoals = self.meta_router._generate_specialized_subgoals(
            goal_id, current_state_high_volatility, tags
        )
        
        # Should have volatility stabilization subgoal
        volatility_subgoals = [s for s in specialized_subgoals if 'volatility_stabilization' in s['subgoal_id']]
        self.assertGreater(len(volatility_subgoals), 0)
        
        # Test with low coherence
        current_state_low_coherence = {
            'contradiction_count': 1,
            'volatility_score': 0.3,
            'coherence_score': 0.3  # Low coherence
        }
        
        specialized_subgoals = self.meta_router._generate_specialized_subgoals(
            goal_id, current_state_low_coherence, tags
        )
        
        # Should have coherence improvement subgoal
        coherence_subgoals = [s for s in specialized_subgoals if 'coherence_improvement' in s['subgoal_id']]
        self.assertGreater(len(coherence_subgoals), 0)
    
    def test_route_single_subgoal(self):
        """Test routing a single subgoal to an agent."""
        # Create mock simulation result
        mock_repairs = [
            SimulatedRepair("belief_clarification", 0.8, 0.1, -0.05, 0.2, 0.9,
                           ["Strategy: belief_clarification"], 30.0, [], 0.85)
        ]
        
        mock_simulation_result = RepairSimulationResult(
            goal_id="test_goal",
            current_state={'drift_type': 'contradiction'},
            simulated_repairs=mock_repairs,
            best_repair=mock_repairs[0],
            reasoning_summary="Test simulation",
            simulation_timestamp=time.time()
        )
        
        # Create test subgoal
        subgoal = {
            'subgoal_id': 'test_goal_subgoal_1',
            'goal_id': 'test_goal',
            'strategy': 'belief_clarification',
            'predicted_score': 0.8,
            'confidence': 0.9,
            'estimated_duration': 30.0,
            'risk_factors': [],
            'reasoning_chain': ["Strategy: belief_clarification"],
            'priority': 0.72,
            'tags': ["test", "strategy:belief_clarification"]
        }
        
        current_state = {
            'drift_type': 'contradiction',
            'drift_score': 0.6,
            'cluster_id': 'test_cluster'
        }
        
        # Mock symbolic rules
        mock_rules = [Mock(confidence=0.8, antecedent="DriftType(contradiction)")]
        self.mock_self_model.query_rules.return_value = mock_rules
        
        result = self.meta_router._route_single_subgoal(subgoal, current_state, mock_simulation_result)
        
        self.assertIsInstance(result, RoutedSubgoal)
        self.assertEqual(result.subgoal_id, subgoal['subgoal_id'])
        self.assertEqual(result.goal_id, subgoal['goal_id'])
        self.assertIn(result.agent_type, self.meta_router.agent_capabilities)
        self.assertEqual(result.priority, subgoal['priority'])
        self.assertIsInstance(result.reasoning_path, list)
        self.assertIsInstance(result.estimated_duration, float)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.dependencies, list)
        self.assertIsInstance(result.tags, list)
    
    def test_calculate_agent_score(self):
        """Test calculating agent scores for subgoal routing."""
        # Create test subgoal
        subgoal = {
            'strategy': 'belief_clarification',
            'estimated_duration': 30.0,
            'tags': ['contradiction_resolution'],
            'reasoning_chain': ['introspection']
        }
        
        current_state = {
            'drift_type': 'contradiction',
            'drift_score': 0.6
        }
        
        mock_simulation_result = Mock()
        
        # Test with reflector agent
        agent_type = 'reflector'
        capabilities = self.meta_router.agent_capabilities[agent_type]
        
        score = self.meta_router._calculate_agent_score(
            agent_type, capabilities, subgoal, current_state, mock_simulation_result
        )
        
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1)
        
        # Test with clarifier agent
        agent_type = 'clarifier'
        capabilities = self.meta_router.agent_capabilities[agent_type]
        
        score = self.meta_router._calculate_agent_score(
            agent_type, capabilities, subgoal, current_state, mock_simulation_result
        )
        
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1)
    
    def test_get_symbolic_routing_bonus(self):
        """Test getting symbolic routing bonus for agents."""
        # Mock symbolic rules
        mock_rules = [
            Mock(confidence=0.8, antecedent="DriftType(contradiction)"),
            Mock(confidence=0.7, antecedent="contradiction_resolution")
        ]
        self.mock_self_model.query_rules.return_value = mock_rules
        
        agent_type = 'reflector'
        subgoal = {
            'tags': ['contradiction_resolution']
        }
        current_state = {
            'drift_type': 'contradiction'
        }
        
        bonus = self.meta_router._get_symbolic_routing_bonus(agent_type, subgoal, current_state)
        
        self.assertGreater(bonus, 0)
        self.assertLessEqual(bonus, 1)
    
    def test_generate_routing_reasoning(self):
        """Test generating reasoning path for routing decision."""
        subgoal = {
            'subgoal_id': 'test_subgoal',
            'strategy': 'belief_clarification'
        }
        
        agent_type = 'reflector'
        capabilities = self.meta_router.agent_capabilities[agent_type]
        
        agent_scores = [
            ('reflector', capabilities, 0.8),
            ('clarifier', self.meta_router.agent_capabilities['clarifier'], 0.6)
        ]
        
        reasoning = self.meta_router._generate_routing_reasoning(
            subgoal, agent_type, capabilities, agent_scores
        )
        
        self.assertIsInstance(reasoning, list)
        self.assertGreater(len(reasoning), 0)
        self.assertIn('test_subgoal', reasoning[0])
        self.assertIn('belief_clarification', reasoning[1])
        self.assertIn('reflector', reasoning[2])
        self.assertIn('0.800', reasoning[3])  # Agent score
    
    def test_calculate_routing_confidence(self):
        """Test calculating routing confidence."""
        agent_score = 0.8
        subgoal = {
            'confidence': 0.9
        }
        capabilities = {
            'strengths': ['introspection', 'belief_analysis', 'contradiction_resolution']
        }
        
        confidence = self.meta_router._calculate_routing_confidence(agent_score, subgoal, capabilities)
        
        self.assertGreater(confidence, 0)
        self.assertLessEqual(confidence, 1)
    
    def test_determine_subgoal_dependencies(self):
        """Test determining dependencies between subgoals."""
        subgoal = {
            'strategy': 'fact_consolidation',
            'goal_id': 'test_goal',
            'tags': []
        }
        
        # Create mock simulation result with belief clarification
        mock_repairs = [
            SimulatedRepair("belief_clarification", 0.8, 0.1, -0.05, 0.2, 0.9,
                           ["Strategy: belief_clarification"], 30.0, [], 0.85)
        ]
        
        mock_simulation_result = RepairSimulationResult(
            goal_id="test_goal",
            current_state={'drift_type': 'contradiction'},
            simulated_repairs=mock_repairs,
            best_repair=mock_repairs[0],
            reasoning_summary="Test simulation",
            simulation_timestamp=time.time()
        )
        
        dependencies = self.meta_router._determine_subgoal_dependencies(subgoal, mock_simulation_result)
        
        self.assertIsInstance(dependencies, list)
        # fact_consolidation should depend on belief_clarification
        self.assertIn('test_goal_belief_clarification', dependencies)
    
    def test_optimize_routing_plan(self):
        """Test optimizing routing plan."""
        # Create test routed subgoals
        subgoals = [
            RoutedSubgoal(
                subgoal_id="test_goal_1",
                goal_id="test_goal",
                agent_type="reflector",
                priority=0.8,
                reasoning_path=["Test reasoning"],
                estimated_duration=30.0,
                confidence=0.9,
                dependencies=[],
                tags=["test"]
            ),
            RoutedSubgoal(
                subgoal_id="test_goal_2",
                goal_id="test_goal",
                agent_type="clarifier",
                priority=0.6,
                reasoning_path=["Test reasoning"],
                estimated_duration=45.0,
                confidence=0.7,
                dependencies=[],
                tags=["test"]
            )
        ]
        
        optimized = self.meta_router._optimize_routing_plan(subgoals)
        
        self.assertIsInstance(optimized, list)
        self.assertEqual(len(optimized), 2)  # Both should be kept as they're different agents
        
        # Test with duplicate agent types
        duplicate_subgoals = [
            RoutedSubgoal(
                subgoal_id="test_goal_1",
                goal_id="test_goal",
                agent_type="reflector",
                priority=0.8,
                reasoning_path=["Test reasoning"],
                estimated_duration=30.0,
                confidence=0.9,
                dependencies=[],
                tags=["test"]
            ),
            RoutedSubgoal(
                subgoal_id="test_goal_2",
                goal_id="test_goal",
                agent_type="reflector",  # Same agent type
                priority=0.6,
                reasoning_path=["Test reasoning"],
                estimated_duration=45.0,
                confidence=0.7,
                dependencies=[],
                tags=["test"]
            )
        ]
        
        optimized = self.meta_router._optimize_routing_plan(duplicate_subgoals)
        
        # Should deduplicate to one reflector
        self.assertEqual(len(optimized), 1)
        self.assertEqual(optimized[0].agent_type, "reflector")
    
    def test_subgoals_are_different(self):
        """Test checking if two subgoals are significantly different."""
        subgoal1 = RoutedSubgoal(
            subgoal_id="test_goal_1",
            goal_id="test_goal",
            agent_type="reflector",
            priority=0.8,
            reasoning_path=["Test reasoning"],
            estimated_duration=30.0,
            confidence=0.9,
            dependencies=[],
            tags=["test", "strategy:belief_clarification"]
        )
        
        subgoal2 = RoutedSubgoal(
            subgoal_id="test_goal_2",
            goal_id="test_goal",
            agent_type="reflector",
            priority=0.6,
            reasoning_path=["Test reasoning"],
            estimated_duration=45.0,
            confidence=0.7,
            dependencies=[],
            tags=["test", "strategy:fact_consolidation"]
        )
        
        # Should be different due to different strategy tags
        self.assertTrue(self.meta_router._subgoals_are_different(subgoal1, subgoal2))
        
        # Test with similar subgoals
        subgoal3 = RoutedSubgoal(
            subgoal_id="test_goal_3",
            goal_id="test_goal",
            agent_type="reflector",
            priority=0.75,  # Similar priority
            reasoning_path=["Test reasoning"],
            estimated_duration=35.0,
            confidence=0.85,
            dependencies=[],
            tags=["test", "strategy:belief_clarification"]  # Same tags
        )
        
        # Should be similar
        self.assertFalse(self.meta_router._subgoals_are_different(subgoal1, subgoal3))
    
    def test_generate_routing_summary(self):
        """Test generating routing summary."""
        # Create test subgoals
        subgoals = [
            RoutedSubgoal(
                subgoal_id="test_goal_1",
                goal_id="test_goal",
                agent_type="reflector",
                priority=0.8,
                reasoning_path=["Test reasoning"],
                estimated_duration=30.0,
                confidence=0.9,
                dependencies=[],
                tags=["test"]
            ),
            RoutedSubgoal(
                subgoal_id="test_goal_2",
                goal_id="test_goal",
                agent_type="clarifier",
                priority=0.6,
                reasoning_path=["Test reasoning"],
                estimated_duration=45.0,
                confidence=0.7,
                dependencies=[],
                tags=["test"]
            )
        ]
        
        mock_simulation_result = Mock()
        mock_simulation_result.best_repair = Mock()
        mock_simulation_result.best_repair.strategy = "belief_clarification"
        mock_simulation_result.best_repair.predicted_score = 0.8
        
        goal_description = "Test goal description"
        
        summary = self.meta_router._generate_routing_summary(
            subgoals, mock_simulation_result, goal_description
        )
        
        self.assertIsInstance(summary, str)
        self.assertIn("2 subgoals", summary)
        self.assertIn("Reflector", summary)
        self.assertIn("Clarifier", summary)
        self.assertIn("belief_clarification", summary)
        self.assertIn("0.800", summary)  # Predicted score


class TestRoutedSubgoal(unittest.TestCase):
    """Test cases for RoutedSubgoal dataclass."""
    
    def test_subgoal_creation(self):
        """Test creating a routed subgoal."""
        subgoal = RoutedSubgoal(
            subgoal_id="test_subgoal",
            goal_id="test_goal",
            agent_type="reflector",
            priority=0.8,
            reasoning_path=["Test reasoning"],
            estimated_duration=30.0,
            confidence=0.9,
            dependencies=["dep1", "dep2"],
            tags=["test", "contradiction_resolution"]
        )
        
        self.assertEqual(subgoal.subgoal_id, "test_subgoal")
        self.assertEqual(subgoal.goal_id, "test_goal")
        self.assertEqual(subgoal.agent_type, "reflector")
        self.assertEqual(subgoal.priority, 0.8)
        self.assertEqual(subgoal.estimated_duration, 30.0)
        self.assertEqual(subgoal.confidence, 0.9)
        self.assertEqual(len(subgoal.dependencies), 2)
        self.assertEqual(len(subgoal.tags), 2)
    
    def test_subgoal_serialization(self):
        """Test subgoal serialization to/from dictionary."""
        subgoal = RoutedSubgoal(
            subgoal_id="test_subgoal",
            goal_id="test_goal",
            agent_type="clarifier",
            priority=0.7,
            reasoning_path=["Test reasoning"],
            estimated_duration=45.0,
            confidence=0.8,
            dependencies=["dep1"],
            tags=["test"]
        )
        
        # Convert to dictionary
        subgoal_dict = subgoal.to_dict()
        
        # Convert back to subgoal
        reconstructed_subgoal = RoutedSubgoal.from_dict(subgoal_dict)
        
        self.assertEqual(subgoal.subgoal_id, reconstructed_subgoal.subgoal_id)
        self.assertEqual(subgoal.goal_id, reconstructed_subgoal.goal_id)
        self.assertEqual(subgoal.agent_type, reconstructed_subgoal.agent_type)
        self.assertEqual(subgoal.priority, reconstructed_subgoal.priority)
        self.assertEqual(subgoal.dependencies, reconstructed_subgoal.dependencies)
        self.assertEqual(subgoal.tags, reconstructed_subgoal.tags)


class TestRoutingPlan(unittest.TestCase):
    """Test cases for RoutingPlan dataclass."""
    
    def test_plan_creation(self):
        """Test creating a routing plan."""
        subgoals = [
            RoutedSubgoal(
                subgoal_id="test_goal_1",
                goal_id="test_goal",
                agent_type="reflector",
                priority=0.8,
                reasoning_path=["Test reasoning"],
                estimated_duration=30.0,
                confidence=0.9,
                dependencies=[],
                tags=["test"]
            )
        ]
        
        plan = RoutingPlan(
            goal_id="test_goal",
            subgoals=subgoals,
            total_estimated_duration=30.0,
            routing_confidence=0.9,
            reasoning_summary="Test routing summary",
            routing_timestamp=time.time()
        )
        
        self.assertEqual(plan.goal_id, "test_goal")
        self.assertEqual(len(plan.subgoals), 1)
        self.assertEqual(plan.total_estimated_duration, 30.0)
        self.assertEqual(plan.routing_confidence, 0.9)
        self.assertEqual(plan.reasoning_summary, "Test routing summary")
    
    def test_plan_serialization(self):
        """Test plan serialization to/from dictionary."""
        subgoals = [
            RoutedSubgoal(
                subgoal_id="test_goal_1",
                goal_id="test_goal",
                agent_type="reflector",
                priority=0.8,
                reasoning_path=["Test reasoning"],
                estimated_duration=30.0,
                confidence=0.9,
                dependencies=[],
                tags=["test"]
            )
        ]
        
        plan = RoutingPlan(
            goal_id="test_goal",
            subgoals=subgoals,
            total_estimated_duration=30.0,
            routing_confidence=0.9,
            reasoning_summary="Test routing summary",
            routing_timestamp=time.time()
        )
        
        # Convert to dictionary
        plan_dict = plan.to_dict()
        
        # Convert back to plan
        reconstructed_plan = RoutingPlan.from_dict(plan_dict)
        
        self.assertEqual(plan.goal_id, reconstructed_plan.goal_id)
        self.assertEqual(len(plan.subgoals), len(reconstructed_plan.subgoals))
        self.assertEqual(plan.total_estimated_duration, reconstructed_plan.total_estimated_duration)
        self.assertEqual(plan.routing_confidence, reconstructed_plan.routing_confidence)
        self.assertEqual(plan.reasoning_summary, reconstructed_plan.reasoning_summary)


class TestGlobalFunctions(unittest.TestCase):
    """Test cases for global functions."""
    
    @patch('agents.meta_router._meta_router_instance', None)
    def test_get_meta_router(self):
        """Test getting the global meta router instance."""
        router = get_meta_router()
        self.assertIsInstance(router, MetaRouter)
    
    @patch('agents.meta_router.get_meta_router')
    def test_route_subgoals(self):
        """Test the convenience function for routing subgoals."""
        mock_router = Mock()
        mock_plan = Mock()
        mock_router.route_subgoals.return_value = mock_plan
        
        with patch('agents.meta_router.get_meta_router', return_value=mock_router):
            result = route_subgoals("test_goal", {'drift_type': 'contradiction'})
            
            self.assertEqual(result, mock_plan)
            mock_router.route_subgoals.assert_called_once()


if __name__ == '__main__':
    unittest.main() 