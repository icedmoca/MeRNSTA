#!/usr/bin/env python3
"""
Test suite for storage/drive_system.py - Phase 7: Motivational Drives & Intent Engine
"""

import pytest
import tempfile
import os
import time
from unittest.mock import Mock, patch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.drive_system import (
    DriveSignal, MotivationalGoal, MotivationalDriveSystem, get_drive_system
)
from agents.base import Goal


class TestDriveSignal:
    """Test DriveSignal functionality."""
    
    def test_drive_signal_creation(self):
        """Test creating a DriveSignal."""
        signal = DriveSignal(
            token_id=123,
            drive_type="curiosity",
            strength=0.8,
            timestamp=time.time(),
            cluster_id="test_cluster",
            source_facts=["fact_1", "fact_2"]
        )
        
        assert signal.token_id == 123
        assert signal.drive_type == "curiosity"
        assert signal.strength == 0.8
        assert signal.cluster_id == "test_cluster"
        assert len(signal.source_facts) == 2
        assert signal.decay_rate == 0.1  # Default value
    
    def test_drive_signal_current_strength(self):
        """Test drive signal strength decay over time."""
        # Create signal from 1 hour ago
        past_time = time.time() - 3600  # 1 hour ago
        signal = DriveSignal(
            token_id=456,
            drive_type="stability",
            strength=1.0,
            timestamp=past_time,
            decay_rate=0.2
        )
        
        # Current strength should be less than original due to decay
        current_strength = signal.get_current_strength()
        assert current_strength < 1.0
        assert current_strength > 0.0
    
    def test_drive_signal_serialization(self):
        """Test DriveSignal to/from dict conversion."""
        signal = DriveSignal(
            token_id=789,
            drive_type="conflict",
            strength=0.6,
            timestamp=time.time()
        )
        
        # Convert to dict
        signal_dict = signal.to_dict()
        assert isinstance(signal_dict, dict)
        assert signal_dict["token_id"] == 789
        assert signal_dict["drive_type"] == "conflict"
        
        # Convert back from dict
        restored_signal = DriveSignal.from_dict(signal_dict)
        assert restored_signal.token_id == signal.token_id
        assert restored_signal.drive_type == signal.drive_type


class TestMotivationalGoal:
    """Test MotivationalGoal functionality."""
    
    def test_motivational_goal_creation(self):
        """Test creating a MotivationalGoal."""
        goal = MotivationalGoal(
            goal_id="test_goal_1",
            description="Test motivational goal",
            priority=0.8,
            strategy="curiosity_exploration",
            driving_motives={"curiosity": 0.9, "novelty": 0.6},
            tension_score=0.75,
            autonomy_level=0.85
        )
        
        assert goal.goal_id == "test_goal_1"
        assert goal.driving_motives["curiosity"] == 0.9
        assert goal.tension_score == 0.75
        assert goal.autonomy_level == 0.85


class TestMotivationalDriveSystem:
    """Test MotivationalDriveSystem functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_drive_system.db")
        self.drive_system = MotivationalDriveSystem(self.db_path)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_drive_system_initialization(self):
        """Test MotivationalDriveSystem initialization."""
        assert self.drive_system.db_path == self.db_path
        assert isinstance(self.drive_system.drive_weights, dict)
        assert "curiosity" in self.drive_system.drive_weights
        assert "coherence" in self.drive_system.drive_weights
        assert "stability" in self.drive_system.drive_weights
        assert self.drive_system.tension_threshold == 0.7
    
    @patch('storage.drive_system.sqlite3.connect')
    def test_evaluate_token_state_no_facts(self, mock_connect):
        """Test evaluating token state with no facts."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        drives = self.drive_system.evaluate_token_state(999)
        
        # Should return zero drives for non-existent token
        assert all(value == 0.0 for value in drives.values())
        assert "curiosity" in drives
        assert "coherence" in drives
        assert "stability" in drives
    
    @patch('storage.drive_system.sqlite3.connect')
    def test_evaluate_token_state_with_facts(self, mock_connect):
        """Test evaluating token state with facts."""
        # Mock database response with sample facts
        mock_facts = [
            (1, "subject1", "predicate1", "object1", 0.8, True, time.time(), 2, 0.6),  # High volatility, contradiction
            (2, "subject2", "predicate2", "object2", 0.3, False, time.time(), 5, 0.9),  # Low volatility, no contradiction
        ]
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = mock_facts
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        drives = self.drive_system.evaluate_token_state(123)
        
        # Should calculate drives based on fact characteristics
        assert isinstance(drives, dict)
        assert all(0.0 <= value <= 1.0 for value in drives.values())
        assert "curiosity" in drives
        assert "coherence" in drives
        assert "stability" in drives
        assert "novelty" in drives
        assert "conflict" in drives
    
    @patch('storage.drive_system.sqlite3.connect')
    def test_compute_drive_tension(self, mock_connect):
        """Test computing drive tension for a cluster."""
        # Mock database to return token IDs for cluster
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [(123,), (456,)]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        # Mock evaluate_token_state to return predictable drives
        with patch.object(self.drive_system, 'evaluate_token_state') as mock_eval:
            mock_eval.side_effect = [
                {"conflict": 0.8, "coherence": 0.3, "curiosity": 0.6, "novelty": 0.4, "stability": 0.7},
                {"conflict": 0.6, "coherence": 0.5, "curiosity": 0.8, "novelty": 0.2, "stability": 0.9}
            ]
            
            tension = self.drive_system.compute_drive_tension("test_cluster")
            
            # Should return a reasonable tension score
            assert isinstance(tension, float)
            assert 0.0 <= tension <= 1.0
    
    @patch('storage.drive_system.sqlite3.connect')
    def test_rank_tokens_by_drive_pressure(self, mock_connect):
        """Test ranking tokens by drive pressure."""
        # Mock database to return token IDs
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [(123,), (456,), (789,)]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        # Mock evaluate_token_state to return different drive levels
        with patch.object(self.drive_system, 'evaluate_token_state') as mock_eval:
            mock_eval.side_effect = [
                {"conflict": 0.9, "coherence": 0.2, "curiosity": 0.8, "novelty": 0.3, "stability": 0.4},  # High pressure
                {"conflict": 0.3, "coherence": 0.8, "curiosity": 0.2, "novelty": 0.1, "stability": 0.9},  # Low pressure
                {"conflict": 0.6, "coherence": 0.5, "curiosity": 0.7, "novelty": 0.5, "stability": 0.6}   # Medium pressure
            ]
            
            ranked_tokens = self.drive_system.rank_tokens_by_drive_pressure(10)
            
            # Should return list of (token_id, pressure) tuples sorted by pressure
            assert isinstance(ranked_tokens, list)
            assert len(ranked_tokens) <= 10
            
            if len(ranked_tokens) > 1:
                # Should be sorted by pressure (highest first)
                assert ranked_tokens[0][1] >= ranked_tokens[1][1]
    
    def test_spawn_goal_if_needed_low_tension(self):
        """Test goal spawning with low tension."""
        # Mock low tension evaluation
        with patch.object(self.drive_system, 'evaluate_token_state') as mock_eval:
            mock_eval.return_value = {"curiosity": 0.2, "coherence": 0.8, "stability": 0.9, "novelty": 0.1, "conflict": 0.0}
            
            goal = self.drive_system.spawn_goal_if_needed(123)
            
            # Should not spawn goal due to low tension
            assert goal is None
    
    def test_spawn_goal_if_needed_high_tension(self):
        """Test goal spawning with high tension."""
        # Mock high tension evaluation
        with patch.object(self.drive_system, 'evaluate_token_state') as mock_eval:
            mock_eval.return_value = {"curiosity": 0.9, "coherence": 0.2, "stability": 0.3, "novelty": 0.8, "conflict": 0.7}
            
            goal = self.drive_system.spawn_goal_if_needed(456)
            
            # Should spawn goal due to high tension
            assert goal is not None
            assert isinstance(goal, MotivationalGoal)
            assert goal.token_id == 456
            assert goal.autonomy_level > 0.5  # Should be autonomous
    
    def test_spawn_goal_forced(self):
        """Test forced goal spawning."""
        # Mock any tension level - should spawn due to force=True
        with patch.object(self.drive_system, 'evaluate_token_state') as mock_eval:
            mock_eval.return_value = {"curiosity": 0.1, "coherence": 0.9, "stability": 0.8, "novelty": 0.0, "conflict": 0.0}
            
            goal = self.drive_system.spawn_goal_if_needed(789, force=True)
            
            # Should spawn goal even with low tension due to force=True
            assert goal is not None
            assert isinstance(goal, MotivationalGoal)
            assert goal.token_id == 789
    
    def test_update_drive_weights(self):
        """Test updating drive weights."""
        original_weights = self.drive_system.drive_weights.copy()
        
        new_weights = {
            "curiosity": 0.95,
            "coherence": 0.85,
            "invalid_drive": 0.5  # Should be ignored
        }
        
        self.drive_system.update_drive_weights(new_weights)
        
        # Valid drives should be updated
        assert self.drive_system.drive_weights["curiosity"] == 0.95
        assert self.drive_system.drive_weights["coherence"] == 0.85
        
        # Invalid drives should be ignored
        assert "invalid_drive" not in self.drive_system.drive_weights
        
        # Other drives should remain unchanged
        assert self.drive_system.drive_weights["stability"] == original_weights["stability"]
    
    def test_get_current_dominant_drives_no_tokens(self):
        """Test getting dominant drives with no tokens."""
        with patch.object(self.drive_system, 'rank_tokens_by_drive_pressure') as mock_rank:
            mock_rank.return_value = []
            
            dominant_drives = self.drive_system.get_current_dominant_drives()
            
            # Should return empty dict when no tokens have pressure
            assert dominant_drives == {}
    
    def test_get_current_dominant_drives_with_tokens(self):
        """Test getting dominant drives with active tokens."""
        with patch.object(self.drive_system, 'rank_tokens_by_drive_pressure') as mock_rank:
            mock_rank.return_value = [(123, 0.8), (456, 0.6)]
            
            with patch.object(self.drive_system, 'evaluate_token_state') as mock_eval:
                mock_eval.side_effect = [
                    {"curiosity": 0.9, "coherence": 0.4, "stability": 0.6, "novelty": 0.3, "conflict": 0.7},
                    {"curiosity": 0.5, "coherence": 0.8, "stability": 0.8, "novelty": 0.2, "conflict": 0.3}
                ]
                
                dominant_drives = self.drive_system.get_current_dominant_drives()
                
                # Should return aggregated drives
                assert isinstance(dominant_drives, dict)
                assert all(drive in dominant_drives for drive in ["curiosity", "coherence", "stability"])
                assert all(0.0 <= strength <= 1.0 for strength in dominant_drives.values())
    
    def test_analyze_drive_trends_no_history(self):
        """Test analyzing drive trends with no history."""
        # Empty drive history
        trends = self.drive_system.analyze_drive_trends()
        
        assert "error" in trends
        assert "No drive history available" in trends["error"]
    
    def test_get_drive_history(self):
        """Test getting drive history."""
        # Initially should be empty
        history = self.drive_system.get_drive_history(24)
        assert history == []


class TestGlobalDriveSystem:
    """Test global drive system functionality."""
    
    def test_get_drive_system_singleton(self):
        """Test that get_drive_system returns singleton instance."""
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_singleton.db")
        
        try:
            # Clear any existing instance
            if hasattr(get_drive_system, '_instance'):
                delattr(get_drive_system, '_instance')
            
            # Get two instances
            system1 = get_drive_system(db_path)
            system2 = get_drive_system(db_path)
            
            # Should be the same instance
            assert system1 is system2
            
        finally:
            # Cleanup
            if hasattr(get_drive_system, '_instance'):
                delattr(get_drive_system, '_instance')
            if os.path.exists(db_path):
                os.remove(db_path)
            os.rmdir(temp_dir)


class TestDriveSystemIntegration:
    """Test drive system integration scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_integration.db")
        self.drive_system = MotivationalDriveSystem(self.db_path)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_full_drive_evaluation_cycle(self):
        """Test complete drive evaluation and goal spawning cycle."""
        # Mock token with high conflict and low coherence
        with patch.object(self.drive_system, 'evaluate_token_state') as mock_eval:
            mock_eval.return_value = {
                "curiosity": 0.6,
                "coherence": 0.2,  # Low coherence
                "stability": 0.4,
                "novelty": 0.7,
                "conflict": 0.9    # High conflict
            }
            
            # Check drive tension
            token_id = 12345
            drives = self.drive_system.evaluate_token_state(token_id)
            
            # Verify drives are as expected
            assert drives["conflict"] == 0.9
            assert drives["coherence"] == 0.2
            
            # Calculate tension manually
            expected_tension = (0.9 * 1.2 + (1.0 - 0.2) * 1.0 + 0.6 * 0.8 + 0.7 * 0.6 - 0.4 * 0.4)
            
            # Try to spawn goal
            goal = self.drive_system.spawn_goal_if_needed(token_id)
            
            if expected_tension >= self.drive_system.tension_threshold:
                assert goal is not None
                assert goal.strategy in ["conflict_resolution", "coherence_repair"]
            else:
                # Force spawn to test goal creation logic
                goal = self.drive_system.spawn_goal_if_needed(token_id, force=True)
                assert goal is not None
            
            # Verify goal properties
            assert goal.driving_motives["conflict"] == 0.9
            assert goal.autonomy_level >= 0.8
    
    def test_drive_system_performance_with_many_tokens(self):
        """Test drive system performance with many tokens."""
        # Mock database with many tokens
        with patch('storage.drive_system.sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            
            # Create 100 mock tokens
            mock_tokens = [(i,) for i in range(100)]
            mock_cursor.fetchall.return_value = mock_tokens
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn
            
            # Mock token evaluation to return reasonable drives
            with patch.object(self.drive_system, 'evaluate_token_state') as mock_eval:
                mock_eval.return_value = {"curiosity": 0.5, "coherence": 0.6, "stability": 0.7, "novelty": 0.3, "conflict": 0.4}
                
                start_time = time.time()
                ranked_tokens = self.drive_system.rank_tokens_by_drive_pressure(50)
                end_time = time.time()
                
                # Should complete within reasonable time (< 1 second for mocked data)
                assert (end_time - start_time) < 1.0
                assert len(ranked_tokens) <= 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])