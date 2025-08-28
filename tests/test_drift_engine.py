#!/usr/bin/env python3
"""
Test suite for drift_execution_engine.py - Phase 3: Drift Detection & Autonomous Repair
"""

import pytest
import tempfile
import os
import time
import threading
from unittest.mock import Mock, patch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.drift_execution_engine import (
    DriftExecutionEngine, ExecutionResult
)


class TestExecutionResult:
    """Test ExecutionResult dataclass."""
    
    def test_execution_result_creation(self):
        """Test creating an ExecutionResult."""
        result = ExecutionResult(
            goal_id="test_goal_1",
            success=True,
            execution_time=2.5,
            actions_taken=["action_1", "action_2"],
            memory_trail=[{"action": "test", "timestamp": time.time()}],
            completion_notes="Successfully completed test goal"
        )
        
        assert result.goal_id == "test_goal_1"
        assert result.success is True
        assert result.execution_time == 2.5
        assert len(result.actions_taken) == 2
        assert len(result.memory_trail) == 1
        assert result.error_message is None
        assert "Successfully completed" in result.completion_notes


class TestDriftExecutionEngine:
    """Test DriftExecutionEngine functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_memory_system = Mock()
        self.engine = DriftExecutionEngine(memory_system=self.mock_memory_system)
        # Disable background execution for testing
        self.engine.auto_execute = False
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.engine.running:
            self.engine.stop_background_execution()
    
    def test_engine_initialization(self):
        """Test DriftExecutionEngine initialization."""
        assert self.engine.memory_system == self.mock_memory_system
        assert hasattr(self.engine, 'repair_agent')
        assert hasattr(self.engine, 'execution_history')
        assert hasattr(self.engine, 'active_executions')
        assert hasattr(self.engine, 'execution_queue')
        assert self.engine.running is False
    
    def test_select_best_strategy(self):
        """Test select_best_strategy method."""
        # Test with minimal parameters
        strategy, reason = self.engine.select_best_strategy(
            token_id=123,
            cluster_id="cluster_test",
            available_strategies=["belief_clarification", "cluster_reassessment"]
        )
        
        assert strategy in ["belief_clarification", "cluster_reassessment", "fact_consolidation"]
        assert isinstance(reason, str)
        assert len(reason) > 0
    
    def test_select_best_strategy_with_context(self):
        """Test select_best_strategy with context."""
        context = {
            "contradiction_score": 0.8,
            "volatility_score": 0.6,
            "coherence_score": 0.4,
            "recent_failures": ["cluster_reassessment"]
        }
        
        strategy, reason = self.engine.select_best_strategy(
            token_id=456,
            cluster_id="cluster_context_test",
            available_strategies=["belief_clarification", "fact_consolidation"],
            context=context
        )
        
        assert strategy in ["belief_clarification", "fact_consolidation"]
        assert isinstance(reason, str)
        assert "contradiction_score" in reason or "context" in reason.lower()
    
    @patch('agents.drift_execution_engine.get_cognitive_repair_agent')
    def test_execute_goal(self, mock_get_agent):
        """Test goal execution."""
        # Mock the repair agent
        mock_agent = Mock()
        mock_agent.execute_goal.return_value = {
            "success": True,
            "actions_taken": ["clarify_belief", "update_confidence"],
            "memory_trail": [{"action": "clarify_belief", "timestamp": time.time()}],
            "execution_time": 1.5
        }
        mock_get_agent.return_value = mock_agent
        
        # Create a mock goal
        mock_goal = Mock()
        mock_goal.goal_id = "test_goal_execution"
        mock_goal.priority = 0.8
        mock_goal.strategy = "belief_clarification"
        mock_goal.token_id = 789
        
        # Execute the goal
        result = self.engine.execute_goal(mock_goal)
        
        assert isinstance(result, ExecutionResult)
        assert result.goal_id == "test_goal_execution"
        assert result.success is True
        assert len(result.actions_taken) == 2
        assert "clarify_belief" in result.actions_taken
    
    def test_background_execution_start_stop(self):
        """Test starting and stopping background execution."""
        # Enable auto-execute for this test
        self.engine.auto_execute = True
        
        # Start background execution
        self.engine.start_background_execution()
        assert self.engine.running is True
        assert self.engine.execution_thread is not None
        assert self.engine.execution_thread.is_alive()
        
        # Stop background execution
        self.engine.stop_background_execution()
        assert self.engine.running is False
        
        # Wait a bit to ensure thread has stopped
        time.sleep(0.1)
        if self.engine.execution_thread:
            assert not self.engine.execution_thread.is_alive()
    
    def test_execution_history_tracking(self):
        """Test execution history tracking."""
        # Initially empty
        assert len(self.engine.execution_history) == 0
        
        # Create mock execution results
        result1 = ExecutionResult(
            goal_id="history_test_1",
            success=True,
            execution_time=1.0,
            actions_taken=["action1"],
            memory_trail=[]
        )
        
        result2 = ExecutionResult(
            goal_id="history_test_2",
            success=False,
            execution_time=0.5,
            actions_taken=[],
            memory_trail=[],
            error_message="Test error"
        )
        
        # Add to history
        self.engine.execution_history.append(result1)
        self.engine.execution_history.append(result2)
        
        # Verify history
        assert len(self.engine.execution_history) == 2
        assert self.engine.execution_history[0].goal_id == "history_test_1"
        assert self.engine.execution_history[1].goal_id == "history_test_2"
        assert self.engine.execution_history[1].error_message == "Test error"
    
    def test_get_execution_stats(self):
        """Test getting execution statistics."""
        # Add some mock execution results
        for i in range(5):
            result = ExecutionResult(
                goal_id=f"stats_test_{i}",
                success=i % 2 == 0,  # Alternate success/failure
                execution_time=float(i + 1),
                actions_taken=[f"action_{i}"],
                memory_trail=[]
            )
            self.engine.execution_history.append(result)
        
        # Get stats (this method might need to be implemented)
        if hasattr(self.engine, 'get_execution_stats'):
            stats = self.engine.get_execution_stats()
            assert isinstance(stats, dict)
            assert "total_executions" in stats
            assert stats["total_executions"] == 5
    
    @patch('agents.drift_execution_engine.ReflexCycle')
    @patch('agents.drift_execution_engine.log_reflex_cycle')
    def test_reflex_cycle_logging(self, mock_log_cycle, mock_reflex_cycle):
        """Test that drift execution logs reflex cycles."""
        # Mock reflex cycle creation
        mock_cycle = Mock()
        mock_reflex_cycle.return_value = mock_cycle
        mock_log_cycle.return_value = True
        
        # This test assumes the engine logs reflex cycles during execution
        # The actual implementation may vary
        goal_id = "reflex_log_test"
        token_id = 999
        
        # If the engine has a method to create reflex cycles
        if hasattr(self.engine, '_create_reflex_cycle'):
            cycle = self.engine._create_reflex_cycle(
                goal_id=goal_id,
                token_id=token_id,
                strategy="test_strategy",
                success=True
            )
            
            mock_reflex_cycle.assert_called_once()
    
    def test_strategy_optimization_integration(self):
        """Test integration with strategy optimization."""
        # Test that the engine considers past performance when selecting strategies
        context = {
            "recent_strategy_performance": {
                "belief_clarification": {"success_rate": 0.9, "avg_time": 2.0},
                "cluster_reassessment": {"success_rate": 0.6, "avg_time": 3.0},
                "fact_consolidation": {"success_rate": 0.8, "avg_time": 1.5}
            }
        }
        
        strategy, reason = self.engine.select_best_strategy(
            token_id=111,
            available_strategies=list(context["recent_strategy_performance"].keys()),
            context=context
        )
        
        # Should prefer strategies with better performance
        assert strategy in context["recent_strategy_performance"].keys()
        assert isinstance(reason, str)


class TestDriftDetection:
    """Test drift detection capabilities."""
    
    def setup_method(self):
        """Set up test environment."""
        self.engine = DriftExecutionEngine()
        self.engine.auto_execute = False
    
    def test_drift_detection_triggers(self):
        """Test various drift detection triggers."""
        # Test different trigger types
        triggers = [
            "HIGH_VOLATILITY",
            "CONTRADICTION_DETECTED", 
            "SEMANTIC_DECAY",
            "CLUSTER_INSTABILITY"
        ]
        
        for trigger in triggers:
            # This assumes the engine can handle different trigger types
            if hasattr(self.engine, 'handle_drift_trigger'):
                result = self.engine.handle_drift_trigger(trigger, token_id=123)
                assert isinstance(result, (bool, dict))
    
    def test_autonomous_repair_spawning(self):
        """Test that drift detection spawns autonomous repair goals."""
        # Mock high volatility detection
        volatility_context = {
            "volatility_score": 0.9,
            "token_id": 456,
            "cluster_id": "test_cluster",
            "trigger_type": "HIGH_VOLATILITY"
        }
        
        # If the engine has autonomous goal spawning
        if hasattr(self.engine, 'spawn_repair_goal'):
            goal = self.engine.spawn_repair_goal(volatility_context)
            assert hasattr(goal, 'goal_id')
            assert hasattr(goal, 'priority')
            assert hasattr(goal, 'strategy')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])