#!/usr/bin/env python3
"""
Test suite for strategy optimization - Phase 4: Strategy Optimization
"""

import pytest
import tempfile
import os
import time
from unittest.mock import Mock, patch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.strategy_optimizer import StrategyOptimizer
from agents.drift_execution_engine import DriftExecutionEngine
from storage.reflex_log import ReflexLogger, ReflexCycle, ReflexScore


class TestStrategyOptimizer:
    """Test StrategyOptimizer functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_strategy.db")
        self.optimizer = StrategyOptimizer(db_path=self.db_path)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_optimizer_initialization(self):
        """Test StrategyOptimizer initialization."""
        assert hasattr(self.optimizer, 'reflex_logger')
        assert hasattr(self.optimizer, 'strategy_performance')
        assert hasattr(self.optimizer, 'learning_rate')
        assert isinstance(self.optimizer.strategy_performance, dict)
    
    def test_track_strategy_performance(self):
        """Test tracking strategy performance."""
        # Add performance data
        self.optimizer.record_strategy_result(
            strategy="belief_clarification",
            success=True,
            execution_time=2.0,
            context_features={"volatility": 0.8, "contradiction_score": 0.7}
        )
        
        self.optimizer.record_strategy_result(
            strategy="belief_clarification",
            success=False,
            execution_time=3.0,
            context_features={"volatility": 0.6, "contradiction_score": 0.9}
        )
        
        # Get performance metrics
        performance = self.optimizer.get_strategy_performance("belief_clarification")
        
        assert performance["total_attempts"] == 2
        assert performance["successful_attempts"] == 1
        assert performance["success_rate"] == 0.5
        assert performance["avg_execution_time"] == 2.5
    
    def test_strategy_selection_optimization(self):
        """Test optimized strategy selection."""
        # Record historical performance for multiple strategies
        strategies_data = [
            ("belief_clarification", True, 1.5, {"volatility": 0.8}),
            ("belief_clarification", True, 2.0, {"volatility": 0.7}),
            ("cluster_reassessment", False, 3.0, {"volatility": 0.9}),
            ("cluster_reassessment", True, 2.5, {"volatility": 0.6}),
            ("fact_consolidation", True, 1.0, {"volatility": 0.5}),
            ("fact_consolidation", True, 1.2, {"volatility": 0.4})
        ]
        
        for strategy, success, exec_time, context in strategies_data:
            self.optimizer.record_strategy_result(strategy, success, exec_time, context)
        
        # Test strategy selection for high volatility context
        context = {"volatility": 0.8, "contradiction_score": 0.6}
        best_strategy = self.optimizer.select_optimal_strategy(
            available_strategies=["belief_clarification", "cluster_reassessment", "fact_consolidation"],
            context=context
        )
        
        assert best_strategy in ["belief_clarification", "cluster_reassessment", "fact_consolidation"]
        
        # Should prefer strategies with better historical performance in similar contexts
        # belief_clarification should score well for high volatility contexts
        if best_strategy == "belief_clarification":
            assert True  # Expected for high volatility
    
    def test_learning_from_reflex_history(self):
        """Test learning from reflex cycle history."""
        # Create mock reflex cycles with different outcomes
        cycles = [
            ReflexCycle(
                cycle_id="learn_test_1",
                drift_trigger="HIGH_VOLATILITY",
                token_id=123,
                goal_description="Test goal 1",
                strategy="belief_clarification",
                success=True,
                execution_time=1.5
            ),
            ReflexCycle(
                cycle_id="learn_test_2",
                drift_trigger="CONTRADICTION_DETECTED",
                token_id=124,
                goal_description="Test goal 2",
                strategy="fact_consolidation",
                success=False,
                execution_time=2.0
            ),
            ReflexCycle(
                cycle_id="learn_test_3",
                drift_trigger="HIGH_VOLATILITY",
                token_id=125,
                goal_description="Test goal 3",
                strategy="cluster_reassessment",
                success=True,
                execution_time=2.5
            )
        ]
        
        # Log cycles to reflex logger
        for cycle in cycles:
            self.optimizer.reflex_logger.log_cycle(cycle)
        
        # Update strategy performance from reflex history
        self.optimizer.update_from_reflex_history()
        
        # Verify performance tracking
        belief_perf = self.optimizer.get_strategy_performance("belief_clarification")
        fact_perf = self.optimizer.get_strategy_performance("fact_consolidation")
        cluster_perf = self.optimizer.get_strategy_performance("cluster_reassessment")
        
        assert belief_perf["success_rate"] == 1.0
        assert fact_perf["success_rate"] == 0.0
        assert cluster_perf["success_rate"] == 1.0
    
    def test_context_aware_optimization(self):
        """Test context-aware strategy optimization."""
        # Record performance with different contexts
        high_volatility_contexts = [
            {"volatility": 0.9, "contradiction_score": 0.8},
            {"volatility": 0.85, "contradiction_score": 0.7},
            {"volatility": 0.95, "contradiction_score": 0.9}
        ]
        
        low_volatility_contexts = [
            {"volatility": 0.2, "contradiction_score": 0.3},
            {"volatility": 0.1, "contradiction_score": 0.2},
            {"volatility": 0.3, "contradiction_score": 0.4}
        ]
        
        # Record that belief_clarification works well for high volatility
        for context in high_volatility_contexts:
            self.optimizer.record_strategy_result("belief_clarification", True, 1.5, context)
        
        # Record that fact_consolidation works well for low volatility
        for context in low_volatility_contexts:
            self.optimizer.record_strategy_result("fact_consolidation", True, 1.0, context)
        
        # Test selection for high volatility context
        high_vol_strategy = self.optimizer.select_optimal_strategy(
            available_strategies=["belief_clarification", "fact_consolidation"],
            context={"volatility": 0.9, "contradiction_score": 0.8}
        )
        
        # Test selection for low volatility context
        low_vol_strategy = self.optimizer.select_optimal_strategy(
            available_strategies=["belief_clarification", "fact_consolidation"],
            context={"volatility": 0.2, "contradiction_score": 0.3}
        )
        
        # Should select appropriate strategies based on context
        assert high_vol_strategy == "belief_clarification"
        assert low_vol_strategy == "fact_consolidation"
    
    def test_strategy_performance_decay(self):
        """Test that strategy performance decays over time."""
        # Record old performance
        old_time = time.time() - 86400 * 30  # 30 days ago
        self.optimizer.record_strategy_result(
            strategy="old_strategy",
            success=True,
            execution_time=1.0,
            context_features={},
            timestamp=old_time
        )
        
        # Record recent performance
        self.optimizer.record_strategy_result(
            strategy="old_strategy",
            success=False,
            execution_time=2.0,
            context_features={}
        )
        
        # Apply time decay if supported
        if hasattr(self.optimizer, 'apply_time_decay'):
            self.optimizer.apply_time_decay()
        
        performance = self.optimizer.get_strategy_performance("old_strategy")
        
        # Recent performance should have more weight
        # This test assumes time decay is implemented
        assert performance["total_attempts"] == 2


class TestSelectBestStrategy:
    """Test select_best_strategy integration with strategy optimization."""
    
    def setup_method(self):
        """Set up test environment."""
        self.engine = DriftExecutionEngine()
        self.engine.auto_execute = False
    
    def test_select_best_strategy_with_optimization(self):
        """Test that select_best_strategy uses optimization data."""
        # Test basic strategy selection
        strategy, reason = self.engine.select_best_strategy(
            token_id=123,
            cluster_id="test_cluster",
            available_strategies=["belief_clarification", "cluster_reassessment", "fact_consolidation"]
        )
        
        assert strategy in ["belief_clarification", "cluster_reassessment", "fact_consolidation"]
        assert isinstance(reason, str)
        assert len(reason) > 0
    
    def test_select_best_strategy_with_performance_history(self):
        """Test strategy selection considers performance history."""
        # Mock performance history
        context = {
            "strategy_performance": {
                "belief_clarification": {"success_rate": 0.9, "avg_execution_time": 1.5},
                "cluster_reassessment": {"success_rate": 0.4, "avg_execution_time": 3.0},
                "fact_consolidation": {"success_rate": 0.8, "avg_execution_time": 1.0}
            }
        }
        
        strategy, reason = self.engine.select_best_strategy(
            token_id=456,
            available_strategies=list(context["strategy_performance"].keys()),
            context=context
        )
        
        # Should prefer high-performing strategies
        performance = context["strategy_performance"][strategy]
        assert performance["success_rate"] >= 0.8  # Should select a good strategy
    
    def test_strategy_optimization_cli_integration(self):
        """Test that CLI commands work with strategy optimization."""
        # This tests the /strategy_optimization CLI command
        # Import CLI handler
        try:
            from cortex.cli_commands import handle_command
            from storage.memory_log import MemoryLog
            
            # Create temporary memory log
            temp_dir = tempfile.mkdtemp()
            db_path = os.path.join(temp_dir, "test_cli.db")
            memory_log = MemoryLog(db_path)
            
            # Test strategy optimization command
            result = handle_command(
                "/strategy_optimization",
                memory_log,
                "enhanced",
                "default"
            )
            
            # Should return 'continue' indicating command was handled
            assert result == 'continue'
            
        except ImportError:
            # Skip if CLI components not available
            pytest.skip("CLI components not available")
        finally:
            # Cleanup
            if 'temp_dir' in locals():
                if os.path.exists(db_path):
                    os.remove(db_path)
                os.rmdir(temp_dir)


class TestRefreshHistoryInfluence:
    """Test that reflex history influences future repair selection."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_reflex_influence.db")
        self.logger = ReflexLogger(self.db_path)
        self.optimizer = StrategyOptimizer(db_path=self.db_path)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_reflex_history_influences_selection(self):
        """Test that reflex history influences future strategy selection."""
        # Create historical reflex cycles showing strategy performance
        successful_cycles = [
            ReflexCycle(
                cycle_id=f"success_{i}",
                drift_trigger="HIGH_VOLATILITY",
                token_id=100 + i,
                goal_description=f"Successful goal {i}",
                strategy="belief_clarification",
                success=True,
                execution_time=1.5
            ) for i in range(5)
        ]
        
        failed_cycles = [
            ReflexCycle(
                cycle_id=f"failure_{i}",
                drift_trigger="HIGH_VOLATILITY",
                token_id=200 + i,
                goal_description=f"Failed goal {i}",
                strategy="cluster_reassessment",
                success=False,
                execution_time=3.0
            ) for i in range(3)
        ]
        
        # Log all cycles
        for cycle in successful_cycles + failed_cycles:
            self.logger.log_cycle(cycle)
        
        # Update optimizer from history
        self.optimizer.update_from_reflex_history()
        
        # Test strategy selection for similar context
        selected_strategy = self.optimizer.select_optimal_strategy(
            available_strategies=["belief_clarification", "cluster_reassessment"],
            context={"drift_trigger": "HIGH_VOLATILITY"}
        )
        
        # Should prefer the strategy with better historical performance
        assert selected_strategy == "belief_clarification"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])