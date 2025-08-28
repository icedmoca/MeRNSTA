#!/usr/bin/env python3
"""
Test suite for reflex_log.py - Phase 2: Reflex Loop + Scoring
"""

import pytest
import tempfile
import os
import time
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.reflex_log import (
    ReflexCycle, ReflexScore, ReflexLogger, log_reflex_cycle, 
    get_reflex_logger, compare_strategies_logically
)


class TestReflexScore:
    """Test ReflexScore functionality."""
    
    def test_reflex_score_creation(self):
        """Test creating a ReflexScore."""
        score = ReflexScore(
            cycle_id="test_cycle_1",
            success=True,
            strategy="belief_clarification",
            token_id=123,
            coherence_delta=0.1,
            volatility_delta=-0.05,
            belief_consistency_delta=0.2,
            score=0.8
        )
        
        assert score.cycle_id == "test_cycle_1"
        assert score.success is True
        assert score.strategy == "belief_clarification"
        assert score.token_id == 123
        assert score.coherence_delta == 0.1
        assert score.score == 0.8
        assert score.affected_facts == []  # Default empty list
    
    def test_reflex_score_post_init(self):
        """Test ReflexScore post-initialization."""
        score = ReflexScore(
            cycle_id="test_cycle_2",
            success=False,
            strategy="cluster_reassessment",
            token_id=456
        )
        
        # Timestamp should be set automatically
        assert score.timestamp > 0
        assert isinstance(score.affected_facts, list)


class TestReflexCycle:
    """Test ReflexCycle functionality."""
    
    def test_reflex_cycle_creation(self):
        """Test creating a ReflexCycle."""
        cycle = ReflexCycle(
            cycle_id="test_cycle_3",
            drift_trigger="HIGH_VOLATILITY",
            token_id=789,
            goal_description="Clarify conflicting beliefs about topic X",
            strategy="belief_clarification",
            success=True,
            execution_time=2.5
        )
        
        assert cycle.cycle_id == "test_cycle_3"
        assert cycle.drift_trigger == "HIGH_VOLATILITY"
        assert cycle.token_id == 789
        assert cycle.success is True
        assert cycle.execution_time == 2.5
        assert cycle.actions_taken == []  # Default empty list
    
    def test_reflex_cycle_dict_conversion(self):
        """Test ReflexCycle to/from dict conversion."""
        cycle = ReflexCycle(
            cycle_id="test_cycle_4",
            drift_trigger="CONTRADICTION_DETECTED",
            token_id=101,
            goal_description="Resolve contradiction",
            strategy="fact_consolidation",
            success=True,
            execution_time=1.2,
            actions_taken=["merge_fact_A", "update_fact_B"]
        )
        
        # Convert to dict
        cycle_dict = cycle.to_dict()
        assert isinstance(cycle_dict, dict)
        assert cycle_dict['cycle_id'] == "test_cycle_4"
        assert cycle_dict['actions_taken'] == ["merge_fact_A", "update_fact_B"]
        
        # Convert back from dict
        restored_cycle = ReflexCycle.from_dict(cycle_dict)
        assert restored_cycle.cycle_id == cycle.cycle_id
        assert restored_cycle.actions_taken == cycle.actions_taken


class TestReflexLogger:
    """Test ReflexLogger functionality."""
    
    def setup_method(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_reflex.db")
        self.logger = ReflexLogger(self.db_path)
    
    def teardown_method(self):
        """Clean up test database."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_log_reflex_cycle(self):
        """Test logging a reflex cycle."""
        cycle = ReflexCycle(
            cycle_id="log_test_1",
            drift_trigger="HIGH_VOLATILITY",
            token_id=123,
            goal_description="Test goal",
            strategy="test_strategy",
            success=True,
            execution_time=1.0
        )
        
        # Log the cycle
        self.logger.log_cycle(cycle)
        
        # Retrieve and verify
        cycles = self.logger.get_cycles_by_token(123)
        assert len(cycles) == 1
        assert cycles[0].cycle_id == "log_test_1"
        assert cycles[0].strategy == "test_strategy"
    
    def test_log_reflex_score(self):
        """Test logging a reflex score."""
        score = ReflexScore(
            cycle_id="score_test_1",
            success=True,
            strategy="belief_clarification",
            token_id=456,
            score=0.75
        )
        
        # Log the score
        self.logger.log_score(score)
        
        # Retrieve and verify
        scores = self.logger.get_scores_by_token(456)
        assert len(scores) == 1
        assert scores[0].cycle_id == "score_test_1"
        assert scores[0].score == 0.75
    
    def test_get_reflex_history(self):
        """Test getting reflex history for a token."""
        # Log multiple cycles
        for i in range(3):
            cycle = ReflexCycle(
                cycle_id=f"history_test_{i}",
                drift_trigger="TEST_TRIGGER",
                token_id=789,
                goal_description=f"Test goal {i}",
                strategy=f"test_strategy_{i}",
                success=i % 2 == 0,  # Alternate success/failure
                execution_time=float(i + 1)
            )
            self.logger.log_cycle(cycle)
            time.sleep(0.01)  # Ensure different timestamps
        
        # Get history
        history = self.logger.get_reflex_history(789)
        assert len(history) == 3
        
        # Should be sorted by timestamp (most recent first)
        timestamps = [cycle.timestamp for cycle in history]
        assert timestamps == sorted(timestamps, reverse=True)
    
    def test_get_strategy_performance(self):
        """Test getting strategy performance metrics."""
        # Log cycles with different strategies
        strategies = ["strategy_A", "strategy_B", "strategy_A"]
        successes = [True, False, True]
        
        for i, (strategy, success) in enumerate(zip(strategies, successes)):
            cycle = ReflexCycle(
                cycle_id=f"perf_test_{i}",
                drift_trigger="TEST_TRIGGER",
                token_id=100 + i,
                goal_description=f"Performance test {i}",
                strategy=strategy,
                success=success,
                execution_time=1.0
            )
            self.logger.log_cycle(cycle)
        
        # Get performance metrics
        performance = self.logger.get_strategy_performance()
        
        # strategy_A: 2 total, 2 successful (100% success rate)
        assert "strategy_A" in performance
        assert performance["strategy_A"]["total_cycles"] == 2
        assert performance["strategy_A"]["successful_cycles"] == 2
        assert performance["strategy_A"]["success_rate"] == 1.0
        
        # strategy_B: 1 total, 0 successful (0% success rate)
        assert "strategy_B" in performance
        assert performance["strategy_B"]["total_cycles"] == 1
        assert performance["strategy_B"]["successful_cycles"] == 0
        assert performance["strategy_B"]["success_rate"] == 0.0


class TestReflexLogging:
    """Test global reflex logging functions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_global_reflex.db")
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_log_reflex_cycle_function(self):
        """Test the global log_reflex_cycle function."""
        cycle = ReflexCycle(
            cycle_id="global_test_1",
            drift_trigger="GLOBAL_TEST",
            token_id=999,
            goal_description="Global logging test",
            strategy="global_strategy",
            success=True,
            execution_time=0.5
        )
        
        # Use custom db_path for testing
        success = log_reflex_cycle(cycle, db_path=self.db_path)
        assert success is True
        
        # Verify logging worked
        logger = ReflexLogger(self.db_path)
        cycles = logger.get_cycles_by_token(999)
        assert len(cycles) == 1
        assert cycles[0].cycle_id == "global_test_1"
    
    def test_get_reflex_logger_function(self):
        """Test the global get_reflex_logger function."""
        logger = get_reflex_logger(db_path=self.db_path)
        assert isinstance(logger, ReflexLogger)
        assert logger.db_path == self.db_path


class TestCompareStrategies:
    """Test logical strategy comparison."""
    
    def test_compare_strategies_logically(self):
        """Test compare_strategies_logically function."""
        strategy_a = "belief_clarification"
        strategy_b = "cluster_reassessment"
        
        # Mock performance data
        performance_data = {
            strategy_a: {"success_rate": 0.8, "avg_execution_time": 2.0},
            strategy_b: {"success_rate": 0.6, "avg_execution_time": 1.5}
        }
        
        # This should return a comparison analysis
        result = compare_strategies_logically(strategy_a, strategy_b, performance_data)
        
        assert isinstance(result, dict)
        assert "comparison" in result
        assert "recommendation" in result
        assert strategy_a in result["comparison"]
        assert strategy_b in result["comparison"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])