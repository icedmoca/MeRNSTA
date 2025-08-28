#!/usr/bin/env python3
"""
Unit tests for Agent Lifecycle Management System (Phase 27)

Tests the autonomous agent lifecycle management based on performance, drift, and alignment.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Import the lifecycle management components
from agents.agent_lifecycle import (
    AgentLifecycleManager, LifecycleAction, LifecycleMetrics, 
    LifecycleDecision
)
from agents.base import BaseAgent
from agents.agent_contract import AgentContract


class MockAgent(BaseAgent):
    """Mock agent for testing."""
    
    def __init__(self, name: str):
        # Bypass the full BaseAgent initialization for testing
        self.name = name
        self.enabled = True
        self.contract = None
        
        # Initialize lifecycle tracking fields
        self.last_promotion = None
        self.last_mutation = None
        self.lifecycle_history = []
        
        # Performance tracking
        self._execution_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._confidence_history = []
        
    def get_agent_instructions(self) -> str:
        return f"Mock instructions for {self.name}"
    
    def respond(self, message: str, context=None):
        return {'success': True, 'message': f'Mock response from {self.name}'}


class TestLifecycleMetrics(unittest.TestCase):
    """Test LifecycleMetrics dataclass."""
    
    def test_lifecycle_metrics_creation(self):
        """Test creating lifecycle metrics."""
        metrics = LifecycleMetrics(
            agent_name="test_agent",
            drift_score=0.3,
            success_rate=0.8,
            execution_count=100
        )
        
        self.assertEqual(metrics.agent_name, "test_agent")
        self.assertEqual(metrics.drift_score, 0.3)
        self.assertEqual(metrics.success_rate, 0.8)
        self.assertEqual(metrics.execution_count, 100)
        self.assertIsInstance(metrics.last_evaluation, datetime)
    
    def test_lifecycle_metrics_defaults(self):
        """Test default values for lifecycle metrics."""
        metrics = LifecycleMetrics(agent_name="test")
        
        self.assertEqual(metrics.drift_score, 0.0)
        self.assertEqual(metrics.success_rate, 0.0)
        self.assertEqual(metrics.execution_count, 0)
        self.assertEqual(metrics.performance_trend, [])


class TestLifecycleDecision(unittest.TestCase):
    """Test LifecycleDecision dataclass."""
    
    def test_lifecycle_decision_creation(self):
        """Test creating lifecycle decisions."""
        decision = LifecycleDecision(
            agent_name="test_agent",
            action=LifecycleAction.PROMOTE,
            reason="High performance",
            confidence=0.9
        )
        
        self.assertEqual(decision.agent_name, "test_agent")
        self.assertEqual(decision.action, LifecycleAction.PROMOTE)
        self.assertEqual(decision.reason, "High performance")
        self.assertEqual(decision.confidence, 0.9)
        self.assertIsInstance(decision.timestamp, datetime)


class TestAgentLifecycleManager(unittest.TestCase):
    """Test the main AgentLifecycleManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock configuration
        self.mock_config = {
            'agent_lifecycle': {
                'enabled': True,
                'thresholds': {
                    'drift_threshold': 0.7,
                    'promotion_threshold': 0.85,
                    'retirement_threshold': 0.3,
                    'mutation_threshold': 0.5
                },
                'min_execution_count': 10,
                'evaluation_window_days': 7,
                'trend_analysis_depth': 20
            }
        }
        
        # Create mock agent with contract
        self.mock_agent = MockAgent("test_agent")
        self.mock_agent.contract = AgentContract(
            agent_name="test_agent",
            purpose="Test agent for lifecycle management",
            capabilities=["testing", "mocking"],
            confidence_vector={
                "testing": 0.8,
                "mocking": 0.7,
                "analysis": 0.6
            }
        )
        
        # Set up some execution history
        self.mock_agent._execution_count = 50
        self.mock_agent._success_count = 40
        self.mock_agent._failure_count = 10
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('agents.agent_lifecycle.get_config')
    @patch('agents.agent_lifecycle.Path')
    def test_lifecycle_manager_initialization(self, mock_path, mock_get_config):
        """Test lifecycle manager initialization."""
        mock_get_config.return_value = self.mock_config
        mock_path.return_value.mkdir = Mock()
        mock_path.return_value.exists = Mock(return_value=False)
        
        manager = AgentLifecycleManager()
        
        self.assertTrue(manager.enabled)
        self.assertEqual(manager.drift_threshold, 0.7)
        self.assertEqual(manager.promotion_threshold, 0.85)
        self.assertEqual(manager.retirement_threshold, 0.3)
        self.assertEqual(manager.mutation_threshold, 0.5)
    
    @patch('agents.agent_lifecycle.get_config')
    def test_evaluate_drift(self, mock_get_config):
        """Test drift evaluation."""
        mock_get_config.return_value = self.mock_config
        
        with patch('agents.agent_lifecycle.Path'):
            manager = AgentLifecycleManager()
        
        # Test with mock agent
        drift_score = manager.evaluate_drift(self.mock_agent)
        
        self.assertIsInstance(drift_score, float)
        self.assertGreaterEqual(drift_score, 0.0)
        self.assertLessEqual(drift_score, 1.0)
    
    @patch('agents.agent_lifecycle.get_config')
    def test_evaluate_drift_no_contract(self, mock_get_config):
        """Test drift evaluation with no contract."""
        mock_get_config.return_value = self.mock_config
        
        with patch('agents.agent_lifecycle.Path'):
            manager = AgentLifecycleManager()
        
        # Agent without contract
        agent_no_contract = MockAgent("no_contract_agent")
        drift_score = manager.evaluate_drift(agent_no_contract)
        
        # Should return default moderate drift
        self.assertEqual(drift_score, 0.5)
    
    @patch('agents.agent_lifecycle.get_config')
    def test_should_promote_high_performance(self, mock_get_config):
        """Test promotion detection for high-performing agent."""
        mock_get_config.return_value = self.mock_config
        
        with patch('agents.agent_lifecycle.Path'):
            manager = AgentLifecycleManager()
        
        # Set up high-performing agent
        self.mock_agent._execution_count = 100
        self.mock_agent._success_count = 90  # 90% success rate
        
        # Mock drift evaluation to return low drift
        with patch.object(manager, 'evaluate_drift', return_value=0.1):
            should_promote = manager.should_promote(self.mock_agent)
        
        self.assertTrue(should_promote)
    
    @patch('agents.agent_lifecycle.get_config')
    def test_should_promote_insufficient_executions(self, mock_get_config):
        """Test promotion rejection for insufficient executions."""
        mock_get_config.return_value = self.mock_config
        
        with patch('agents.agent_lifecycle.Path'):
            manager = AgentLifecycleManager()
        
        # Set up agent with insufficient executions
        self.mock_agent._execution_count = 5  # Below min_execution_count
        self.mock_agent._success_count = 5
        
        should_promote = manager.should_promote(self.mock_agent)
        
        self.assertFalse(should_promote)
    
    @patch('agents.agent_lifecycle.get_config')
    def test_should_mutate_high_drift(self, mock_get_config):
        """Test mutation detection for high-drift agent."""
        mock_get_config.return_value = self.mock_config
        
        with patch('agents.agent_lifecycle.Path'):
            manager = AgentLifecycleManager()
        
        # Set up agent with sufficient executions
        self.mock_agent._execution_count = 50
        
        # Mock high drift
        with patch.object(manager, 'evaluate_drift', return_value=0.8):  # Above 0.7 threshold
            should_mutate = manager.should_mutate(self.mock_agent)
        
        self.assertTrue(should_mutate)
    
    @patch('agents.agent_lifecycle.get_config')
    def test_should_mutate_low_performance(self, mock_get_config):
        """Test mutation detection for low-performing agent."""
        mock_get_config.return_value = self.mock_config
        
        with patch('agents.agent_lifecycle.Path'):
            manager = AgentLifecycleManager()
        
        # Set up low-performing agent
        self.mock_agent._execution_count = 50
        self.mock_agent._success_count = 20  # 40% success rate (below 50% threshold)
        
        # Mock low drift
        with patch.object(manager, 'evaluate_drift', return_value=0.2):
            should_mutate = manager.should_mutate(self.mock_agent)
        
        self.assertTrue(should_mutate)
    
    @patch('agents.agent_lifecycle.get_config')
    def test_should_retire_poor_performance(self, mock_get_config):
        """Test retirement detection for very poor performing agent."""
        mock_get_config.return_value = self.mock_config
        
        with patch('agents.agent_lifecycle.Path'):
            manager = AgentLifecycleManager()
        
        # Set up very poor performing agent
        self.mock_agent._execution_count = 100  # Sufficient executions
        self.mock_agent._success_count = 25   # 25% success rate (below 30% threshold)
        
        should_retire = manager.should_retire(self.mock_agent)
        
        self.assertTrue(should_retire)
    
    @patch('agents.agent_lifecycle.get_config')
    def test_should_retire_extreme_drift(self, mock_get_config):
        """Test retirement detection for extremely drifted agent."""
        mock_get_config.return_value = self.mock_config
        
        with patch('agents.agent_lifecycle.Path'):
            manager = AgentLifecycleManager()
        
        # Set up agent with sufficient executions
        self.mock_agent._execution_count = 100
        
        # Mock extreme drift
        with patch.object(manager, 'evaluate_drift', return_value=0.95):  # Above 0.9
            should_retire = manager.should_retire(self.mock_agent)
        
        self.assertTrue(should_retire)
    
    @patch('agents.agent_lifecycle.get_config')
    def test_apply_lifecycle_decision_promotion(self, mock_get_config):
        """Test applying lifecycle decision for promotion."""
        mock_get_config.return_value = self.mock_config
        
        with patch('agents.agent_lifecycle.Path'):
            manager = AgentLifecycleManager()
        
        # Set up for promotion
        self.mock_agent._execution_count = 100
        self.mock_agent._success_count = 90
        
        with patch.object(manager, 'evaluate_drift', return_value=0.1):
            with patch.object(manager, '_log_lifecycle_decision'):
                decision = manager.apply_lifecycle_decision(self.mock_agent)
        
        self.assertEqual(decision.action, LifecycleAction.PROMOTE)
        self.assertEqual(decision.agent_name, "test_agent")
        self.assertIn("High performance", decision.reason)
    
    @patch('agents.agent_lifecycle.get_config')
    def test_apply_lifecycle_decision_mutation(self, mock_get_config):
        """Test applying lifecycle decision for mutation."""
        mock_get_config.return_value = self.mock_config
        
        with patch('agents.agent_lifecycle.Path'):
            manager = AgentLifecycleManager()
        
        # Set up for mutation (low performance)
        self.mock_agent._execution_count = 50
        self.mock_agent._success_count = 20  # 40% success rate
        
        with patch.object(manager, 'evaluate_drift', return_value=0.3):
            with patch.object(manager, '_log_lifecycle_decision'):
                decision = manager.apply_lifecycle_decision(self.mock_agent)
        
        self.assertEqual(decision.action, LifecycleAction.MUTATE)
        self.assertEqual(decision.agent_name, "test_agent")
    
    @patch('agents.agent_lifecycle.get_config')
    def test_vector_drift_calculation(self, mock_get_config):
        """Test vector drift calculation."""
        mock_get_config.return_value = self.mock_config
        
        with patch('agents.agent_lifecycle.Path'):
            manager = AgentLifecycleManager()
        
        # Test identical vectors (no drift)
        original = {"capability1": 0.8, "capability2": 0.7}
        current = {"capability1": 0.8, "capability2": 0.7}
        
        drift = manager._calculate_vector_drift(original, current)
        self.assertAlmostEqual(drift, 0.0, places=3)
        
        # Test completely different vectors (maximum drift)
        original = {"capability1": 1.0, "capability2": 1.0}
        current = {"capability1": 0.0, "capability2": 0.0}
        
        drift = manager._calculate_vector_drift(original, current)
        self.assertAlmostEqual(drift, 1.0, places=3)
        
        # Test no common keys (should return max drift)
        original = {"capability1": 0.8}
        current = {"capability2": 0.7}
        
        drift = manager._calculate_vector_drift(original, current)
        self.assertEqual(drift, 1.0)


class TestBaseAgentLifecycleIntegration(unittest.TestCase):
    """Test BaseAgent lifecycle integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_agent = MockAgent("lifecycle_test_agent")
    
    def test_record_execution_success(self):
        """Test recording successful execution."""
        initial_count = self.mock_agent._execution_count
        initial_success = self.mock_agent._success_count
        
        self.mock_agent.record_execution(True, {"task": "test_task"})
        
        self.assertEqual(self.mock_agent._execution_count, initial_count + 1)
        self.assertEqual(self.mock_agent._success_count, initial_success + 1)
        self.assertEqual(len(self.mock_agent.lifecycle_history), 1)
        
        history_entry = self.mock_agent.lifecycle_history[0]
        self.assertTrue(history_entry['success'])
        self.assertEqual(history_entry['context']['task'], "test_task")
    
    def test_record_execution_failure(self):
        """Test recording failed execution."""
        initial_count = self.mock_agent._execution_count
        initial_failure = self.mock_agent._failure_count
        
        self.mock_agent.record_execution(False, {"error": "test_error"})
        
        self.assertEqual(self.mock_agent._execution_count, initial_count + 1)
        self.assertEqual(self.mock_agent._failure_count, initial_failure + 1)
        self.assertEqual(len(self.mock_agent.lifecycle_history), 1)
        
        history_entry = self.mock_agent.lifecycle_history[0]
        self.assertFalse(history_entry['success'])
        self.assertEqual(history_entry['context']['error'], "test_error")
    
    def test_update_lifecycle_event_promotion(self):
        """Test updating lifecycle event for promotion."""
        self.mock_agent.update_lifecycle_event('promotion', {"reason": "high_performance"})
        
        self.assertIsNotNone(self.mock_agent.last_promotion)
        self.assertEqual(len(self.mock_agent.lifecycle_history), 1)
        
        history_entry = self.mock_agent.lifecycle_history[0]
        self.assertEqual(history_entry['event_type'], 'promotion')
        self.assertEqual(history_entry['details']['reason'], 'high_performance')
    
    def test_update_lifecycle_event_mutation(self):
        """Test updating lifecycle event for mutation."""
        self.mock_agent.update_lifecycle_event('mutation', {"reason": "drift_detected"})
        
        self.assertIsNotNone(self.mock_agent.last_mutation)
        self.assertEqual(len(self.mock_agent.lifecycle_history), 1)
        
        history_entry = self.mock_agent.lifecycle_history[0]
        self.assertEqual(history_entry['event_type'], 'mutation')
        self.assertEqual(history_entry['details']['reason'], 'drift_detected')
    
    def test_get_lifecycle_metrics(self):
        """Test getting lifecycle metrics from agent."""
        # Set up some data
        self.mock_agent._execution_count = 100
        self.mock_agent._success_count = 80
        self.mock_agent._failure_count = 20
        
        # Create a contract
        self.mock_agent.contract = AgentContract(
            agent_name="lifecycle_test_agent",
            purpose="Testing lifecycle metrics",
            confidence_vector={"testing": 0.8, "analysis": 0.6}
        )
        
        metrics = self.mock_agent.get_lifecycle_metrics()
        
        self.assertEqual(metrics['execution_count'], 100)
        self.assertEqual(metrics['success_count'], 80)
        self.assertEqual(metrics['failure_count'], 20)
        self.assertEqual(metrics['success_rate'], 0.8)
        self.assertEqual(metrics['agent_name'], "lifecycle_test_agent")
        self.assertIn('testing', metrics['confidence_vector'])
        self.assertIn('analysis', metrics['confidence_vector'])
    
    def test_lifecycle_history_limit(self):
        """Test that lifecycle history is limited to prevent memory bloat."""
        # Add more than 100 entries
        for i in range(150):
            self.mock_agent.record_execution(True, {"execution": i})
        
        # Should be limited to 100 entries
        self.assertEqual(len(self.mock_agent.lifecycle_history), 100)
        
        # Should keep the most recent entries
        last_entry = self.mock_agent.lifecycle_history[-1]
        self.assertEqual(last_entry['context']['execution'], 149)


class TestReflectionOrchestratorIntegration(unittest.TestCase):
    """Test integration with ReflectionOrchestrator."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_agent = MockAgent("reflection_test_agent")
        
    @patch('agents.reflection_orchestrator.AgentLifecycleManager')
    @patch('agents.reflection_orchestrator.get_agent_registry')
    def test_evaluate_agent_lifecycle_after_failure(self, mock_registry, mock_lifecycle_manager):
        """Test lifecycle evaluation after failure in reflection orchestrator."""
        from agents.reflection_orchestrator import ReflectionOrchestrator
        
        # Mock the registry and lifecycle manager
        mock_registry.return_value.get_agent.return_value = self.mock_agent
        mock_manager_instance = Mock()
        mock_lifecycle_manager.return_value = mock_manager_instance
        
        # Mock the decision
        mock_decision = Mock()
        mock_decision.action.value = 'mutate'
        mock_decision.reason = 'High drift detected'
        mock_manager_instance.apply_lifecycle_decision.return_value = mock_decision
        
        # Create orchestrator (with mocked components)
        with patch('agents.reflection_orchestrator.get_config'), \
             patch('agents.reflection_orchestrator.Path'):
            orchestrator = ReflectionOrchestrator()
            orchestrator.agent_replicator = Mock()
            
            # Test the lifecycle evaluation
            orchestrator.evaluate_agent_lifecycle_after_failure(
                "reflection_test_agent", 
                {"error": "contradiction_detected"}
            )
        
        # Verify lifecycle manager was called
        mock_lifecycle_manager.assert_called_once()
        mock_manager_instance.apply_lifecycle_decision.assert_called_once_with(self.mock_agent)
        
        # Verify agent recorded the failure
        self.assertEqual(self.mock_agent._execution_count, 1)
        self.assertEqual(self.mock_agent._failure_count, 1)


class TestCLICommands(unittest.TestCase):
    """Test CLI command functions."""
    
    @patch('cortex.cli_commands.get_agent_replicator')
    def test_promote_agent_command_success(self, mock_get_replicator):
        """Test successful agent promotion via CLI."""
        from cortex.cli_commands import promote_agent_command
        
        # Mock successful promotion
        mock_replicator = Mock()
        mock_replicator.promote_agent.return_value = {
            'success': True,
            'agent_name': 'test_agent',
            'reason': 'Manual promotion via CLI',
            'original_confidence': {'testing': 0.8},
            'confidence_boost': {'testing': 0.88}
        }
        mock_get_replicator.return_value = mock_replicator
        
        result = promote_agent_command('test_agent')
        
        self.assertIn('Successfully promoted', result)
        self.assertIn('test_agent', result)
        self.assertIn('0.800 → 0.880', result)
        mock_replicator.promote_agent.assert_called_once_with('test_agent', 'Manual promotion via CLI')
    
    @patch('cortex.cli_commands.get_agent_replicator')
    def test_retire_agent_command_success(self, mock_get_replicator):
        """Test successful agent retirement via CLI."""
        from cortex.cli_commands import retire_agent_command
        
        # Mock successful retirement
        mock_replicator = Mock()
        mock_replicator.retire_agent.return_value = {
            'success': True,
            'agent_name': 'test_agent',
            'reason': 'Manual retirement via CLI',
            'archived': True
        }
        mock_get_replicator.return_value = mock_replicator
        
        result = retire_agent_command('test_agent')
        
        self.assertIn('Successfully retired', result)
        self.assertIn('test_agent', result)
        self.assertIn('Contract archived', result)
        mock_replicator.retire_agent.assert_called_once_with('test_agent', 'Manual retirement via CLI')
    
    @patch('cortex.cli_commands.AgentLifecycleManager')
    @patch('cortex.cli_commands.get_agent_registry')
    def test_lifecycle_status_command(self, mock_registry, mock_lifecycle_manager):
        """Test lifecycle status command."""
        from cortex.cli_commands import lifecycle_status_command
        
        # Mock agent with metrics
        mock_agent = Mock()
        mock_agent.get_lifecycle_metrics.return_value = {
            'execution_count': 100,
            'success_rate': 0.85,
            'current_performance': 0.88,
            'confidence_variance': 0.02,
            'lifecycle_history_length': 25
        }
        mock_agent.contract = Mock()
        mock_agent.contract.version = "1.0.0"
        mock_agent.contract.purpose = "Test agent for lifecycle status testing"
        mock_agent.contract.capabilities = ["testing", "mocking"]
        mock_agent.contract.confidence_vector = {"testing": 0.8}
        mock_agent.last_promotion = None
        mock_agent.last_mutation = None
        
        mock_registry.return_value.get_agent.return_value = mock_agent
        
        # Mock lifecycle manager
        mock_manager = Mock()
        mock_manager.evaluate_drift.return_value = 0.15
        mock_manager.should_promote.return_value = True
        mock_lifecycle_manager.return_value = mock_manager
        
        result = lifecycle_status_command('test_agent')
        
        self.assertIn('Lifecycle Status for test_agent', result)
        self.assertIn('Execution Count: 100', result)
        self.assertIn('Success Rate: 0.850', result)
        self.assertIn('Drift Score: 0.150', result)
        self.assertIn('🎉 PROMOTE', result)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)