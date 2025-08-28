#!/usr/bin/env python3
"""
Tests for PersonalityEvolver - Phase 29 Memory-Driven Personality Shifts

Tests the comprehensive personality evolution system including:
- Trait shifting logic with configurable bounds
- Memory trend analysis and evolution triggers  
- Contradiction stress detection and personality adaptation
- Weekly cadence and conflict-driven evolution
- CLI command integration and persistence
"""

import pytest
import json
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Test imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.personality_evolver import (
    PersonalityEvolver, PersonalityVector, PersonalityTraitChange,
    PersonalityEvolutionTrace, get_personality_evolver
)


class TestPersonalityVector:
    """Test the PersonalityVector data structure."""
    
    def test_default_initialization(self):
        """Test that PersonalityVector initializes with balanced defaults."""
        vector = PersonalityVector()
        
        # Check all traits default to 0.5 (balanced)
        expected_traits = [
            'curiosity', 'caution', 'empathy', 'assertiveness', 'optimism',
            'analytical', 'creativity', 'confidence', 'skepticism', 'emotional_sensitivity'
        ]
        
        for trait in expected_traits:
            assert hasattr(vector, trait)
            assert getattr(vector, trait) == 0.5
        
        # Check tone and emotional defaults
        assert vector.base_tone == "balanced"
        assert vector.emotional_baseline == "stable"
        assert vector.emotional_reactivity == 0.5
        assert vector.tone_variance == 0.3
        
        # Check evolution metadata
        assert vector.evolution_count == 0
        assert vector.total_evolution_magnitude == 0.0
    
    def test_trait_bounds_checking(self):
        """Test that trait updates respect bounds (0.0 to 1.0)."""
        vector = PersonalityVector()
        
        # Test valid updates
        assert vector.update_trait('curiosity', 0.8) == True
        assert vector.curiosity == 0.8
        
        assert vector.update_trait('empathy', 0.0) == True
        assert vector.empathy == 0.0
        
        assert vector.update_trait('confidence', 1.0) == True
        assert vector.confidence == 1.0
        
        # Test bounds enforcement
        assert vector.update_trait('skepticism', 1.5) == True
        assert vector.skepticism == 1.0  # Clamped to upper bound
        
        assert vector.update_trait('analytical', -0.3) == True
        assert vector.analytical == 0.0  # Clamped to lower bound
        
        # Test invalid trait name
        assert vector.update_trait('nonexistent_trait', 0.5) == False
    
    def test_get_trait_dict(self):
        """Test trait dictionary retrieval."""
        vector = PersonalityVector()
        vector.curiosity = 0.8
        vector.empathy = 0.3
        
        trait_dict = vector.get_trait_dict()
        
        assert isinstance(trait_dict, dict)
        assert len(trait_dict) == 10  # All 10 traits
        assert trait_dict['curiosity'] == 0.8
        assert trait_dict['empathy'] == 0.3
        assert 'skepticism' in trait_dict


class TestPersonalityEvolver:
    """Test the main PersonalityEvolver functionality."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for testing."""
        temp_dir = tempfile.mkdtemp()
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(exist_ok=True)
        
        yield output_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            'personality_evolution': {
                'enabled': True,
                'sensitivity_threshold': 0.3,
                'max_shift_rate': 0.2,
                'mood_decay_rate': 0.1,
                'trait_bounds': {'min': 0.05, 'max': 0.95},
                'weekly_cadence_hours': 168,
                'major_conflict_threshold': 0.8,
                'enable_short_term_evolution': True,
                'enable_long_term_evolution': True,
                'enable_automatic_triggers': True
            }
        }
    
    @pytest.fixture
    def evolver(self, temp_output_dir, mock_config):
        """Create PersonalityEvolver instance for testing."""
        with patch('agents.personality_evolver.get_config', return_value=mock_config):
            with patch('agents.personality_evolver.Path') as mock_path:
                # Mock the file paths to use temp directory
                def side_effect(path_str):
                    if "personality_state.json" in path_str:
                        return temp_output_dir / "personality_state.json"
                    elif "personality_evolution.jsonl" in path_str:
                        return temp_output_dir / "personality_evolution.jsonl"
                    return Path(path_str)
                
                mock_path.side_effect = side_effect
                
                evolver = PersonalityEvolver()
                evolver.state_file = temp_output_dir / "personality_state.json"
                evolver.history_file = temp_output_dir / "personality_evolution.jsonl"
                
                return evolver
    
    def test_initialization(self, evolver, mock_config):
        """Test PersonalityEvolver initialization."""
        assert evolver.name == "personality_evolver"
        assert evolver.sensitivity_threshold == 0.3
        assert evolver.max_shift_rate == 0.2
        assert evolver.trait_bounds == {'min': 0.05, 'max': 0.95}
        
        # Check personality vector is initialized
        assert isinstance(evolver.personality_vector, PersonalityVector)
        assert evolver.personality_vector.curiosity == 0.5
        
        # Check state tracking
        assert isinstance(evolver.evolution_history, list)
        assert len(evolver.evolution_history) == 0
    
    def test_config_loading_with_defaults(self, temp_output_dir):
        """Test that missing config values use defaults."""
        empty_config = {'personality_evolution': {}}
        
        with patch('agents.personality_evolver.get_config', return_value=empty_config):
            with patch('agents.personality_evolver.Path') as mock_path:
                mock_path.side_effect = lambda x: temp_output_dir / "test"
                
                evolver = PersonalityEvolver()
                
                # Should use defaults when config missing [[memory:4199483]]
                assert evolver.sensitivity_threshold > 0  # Has some default
                assert evolver.max_shift_rate > 0
                assert evolver.trait_bounds['min'] >= 0.0
                assert evolver.trait_bounds['max'] <= 1.0
    
    def test_memory_trend_analysis(self, evolver):
        """Test memory trend analysis functionality."""
        # Mock memory system with test data
        mock_memory = Mock()
        mock_facts = [
            Mock(timestamp=time.time() - 3600, subject="work", predicate="feels", object="stressful", confidence=0.8),
            Mock(timestamp=time.time() - 7200, subject="user", predicate="expresses", object="frustration", confidence=0.9),
            Mock(timestamp=time.time() - 10800, subject="project", predicate="requires", object="attention", confidence=0.7)
        ]
        mock_memory.get_facts.return_value = mock_facts
        evolver._memory_system = mock_memory
        
        analysis = evolver.analyze_memory_trends(lookback_days=1)
        
        assert isinstance(analysis, dict)
        assert 'themes' in analysis
        assert 'emotional_drift' in analysis
        assert 'volatility_score' in analysis
        assert 'memory_volume' in analysis
        
        # Should detect some patterns from mock data
        assert analysis['memory_volume'] == 3
        assert len(analysis['themes']) > 0
    
    def test_contradiction_stress_analysis(self, evolver):
        """Test contradiction stress analysis."""
        # Mock dissonance tracker
        mock_dissonance = Mock()
        mock_region = Mock()
        mock_region.pressure_vector = Mock()
        mock_region.pressure_vector.urgency = 0.8
        mock_region.semantic_cluster = "belief_conflict"
        
        mock_dissonance.dissonance_regions = {"region1": mock_region}
        evolver._dissonance_tracker = mock_dissonance
        
        analysis = evolver.analyze_contradiction_stress()
        
        assert isinstance(analysis, dict)
        assert 'pressure_level' in analysis
        assert 'unresolved_conflicts' in analysis
        assert 'stress_indicators' in analysis
        
        # Should detect pressure from mock data
        assert analysis['unresolved_conflicts'] == 1
        assert analysis['pressure_level'] == 0.8
        assert 'high_dissonance_pressure' in analysis['stress_indicators']
    
    def test_evolution_pressure_calculation(self, evolver):
        """Test evolution pressure calculation from multiple sources."""
        memory_analysis = {'volatility_score': 0.4, 'emotional_drift': 0.3}
        contradiction_analysis = {'pressure_level': 0.7}
        belief_analysis = {'change_magnitude': 0.2}
        
        pressure = evolver.calculate_evolution_pressure(
            memory_analysis, contradiction_analysis, belief_analysis
        )
        
        assert isinstance(pressure, float)
        assert 0.0 <= pressure <= 1.0
        
        # High input values should result in significant pressure
        assert pressure > 0.3  # Should be substantial given the input values
    
    def test_trait_adjustment_generation(self, evolver):
        """Test generation of trait adjustments based on analysis."""
        memory_analysis = {
            'emotional_drift': 0.4,  # Should increase empathy
            'volatility_score': 0.3
        }
        contradiction_analysis = {
            'pressure_level': 0.8,  # Should increase skepticism
            'resolution_success_rate': 0.3  # Should decrease confidence
        }
        belief_analysis = {
            'trend_direction': 'volatile',  # Should increase caution
            'change_magnitude': 0.5
        }
        evolution_pressure = 0.6
        
        changes = evolver.generate_trait_adjustments(
            memory_analysis, contradiction_analysis, belief_analysis, evolution_pressure
        )
        
        assert isinstance(changes, list)
        assert len(changes) > 0
        
        # Check that we get PersonalityTraitChange objects
        for change in changes:
            assert isinstance(change, PersonalityTraitChange)
            assert hasattr(change, 'trait_name')
            assert hasattr(change, 'old_value')
            assert hasattr(change, 'new_value')
            assert hasattr(change, 'trigger_type')
            
            # Validate bounds [[memory:4199483]]
            assert 0.0 <= change.new_value <= 1.0
            assert change.new_value >= evolver.trait_bounds['min']
            assert change.new_value <= evolver.trait_bounds['max']
    
    def test_trait_bounds_enforcement(self, evolver):
        """Test that trait changes respect configured bounds."""
        # Force extreme trait values to test bounds
        evolver.personality_vector.confidence = 0.1  # Near minimum
        evolver.personality_vector.skepticism = 0.9  # Near maximum
        
        # Create changes that would violate bounds
        extreme_changes = [
            PersonalityTraitChange(
                trait_name='confidence',
                old_value=0.1,
                new_value=-0.2,  # Would violate min bound
                change_magnitude=0.3,
                trigger_type='test',
                trigger_details='test',
                confidence=1.0,
                timestamp=time.time()
            ),
            PersonalityTraitChange(
                trait_name='skepticism',
                old_value=0.9,
                new_value=1.3,  # Would violate max bound
                change_magnitude=0.4,
                trigger_type='test',
                trigger_details='test',
                confidence=1.0,
                timestamp=time.time()
            )
        ]
        
        changes_applied = evolver.apply_trait_changes(extreme_changes)
        
        # Bounds should be enforced
        assert evolver.personality_vector.confidence >= evolver.trait_bounds['min']
        assert evolver.personality_vector.skepticism <= evolver.trait_bounds['max']
        
        # Changes should still be recorded as applied
        assert 'confidence' in changes_applied
        assert 'skepticism' in changes_applied
    
    def test_weekly_cadence_trigger(self, evolver):
        """Test weekly cadence evolution trigger."""
        # Set last check to more than a week ago
        evolver.last_weekly_check = time.time() - (8 * 24 * 3600)  # 8 days ago
        
        should_trigger = evolver.check_weekly_cadence()
        assert should_trigger == True
        
        # Should update last check time
        assert evolver.last_weekly_check > time.time() - 100  # Recent
    
    def test_major_conflict_trigger(self, evolver):
        """Test major conflict evolution trigger."""
        # Mock dissonance tracker with high pressure
        mock_dissonance = Mock()
        mock_dissonance.analyze_contradiction_stress.return_value = {
            'pressure_level': 0.9  # Above major_conflict_threshold
        }
        evolver._dissonance_tracker = mock_dissonance
        
        with patch.object(evolver, 'analyze_contradiction_stress', return_value={'pressure_level': 0.9}):
            should_trigger = evolver.check_major_conflict_trigger()
            assert should_trigger == True
    
    def test_complete_evolution_cycle(self, evolver):
        """Test a complete personality evolution cycle."""
        # Mock the analysis methods to return evolution-worthy data
        with patch.object(evolver, 'analyze_memory_trends') as mock_memory:
            with patch.object(evolver, 'analyze_contradiction_stress') as mock_stress:
                with patch.object(evolver, 'detect_belief_changes') as mock_beliefs:
                    
                    mock_memory.return_value = {
                        'emotional_drift': 0.4,
                        'volatility_score': 0.3,
                        'themes': [{'theme': 'work', 'frequency': 5}],
                        'memory_volume': 10
                    }
                    
                    mock_stress.return_value = {
                        'pressure_level': 0.8,
                        'unresolved_conflicts': 3,
                        'stress_indicators': ['high_pressure'],
                        'resolution_success_rate': 0.3
                    }
                    
                    mock_beliefs.return_value = {
                        'change_magnitude': 0.5,
                        'trend_direction': 'volatile',
                        'affected_areas': ['work', 'relationships']
                    }
                    
                    # Run evolution
                    trace = evolver.evolve_personality(trigger_type="test", lookback_days=7)
                    
                    assert trace is not None
                    assert isinstance(trace, PersonalityEvolutionTrace)
                    assert trace.trigger_type == "test"
                    assert len(trace.trait_changes) > 0
                    assert trace.evolution_magnitude > 0
                    assert len(trace.natural_language_summary) > 0
    
    def test_evolution_below_threshold(self, evolver):
        """Test that evolution doesn't occur when pressure is below threshold."""
        # Mock low-pressure analyses
        with patch.object(evolver, 'analyze_memory_trends') as mock_memory:
            with patch.object(evolver, 'analyze_contradiction_stress') as mock_stress:
                with patch.object(evolver, 'detect_belief_changes') as mock_beliefs:
                    
                    mock_memory.return_value = {
                        'emotional_drift': 0.1,
                        'volatility_score': 0.1,
                        'themes': [],
                        'memory_volume': 2
                    }
                    
                    mock_stress.return_value = {
                        'pressure_level': 0.1,
                        'unresolved_conflicts': 0,
                        'stress_indicators': []
                    }
                    
                    mock_beliefs.return_value = {
                        'change_magnitude': 0.1,
                        'trend_direction': 'stable'
                    }
                    
                    # Should not trigger evolution (non-manual)
                    trace = evolver.evolve_personality(trigger_type="automatic", lookback_days=7)
                    assert trace is None
    
    def test_personality_status_reporting(self, evolver):
        """Test personality status reporting functionality."""
        # Set some test evolution metadata
        evolver.personality_vector.evolution_count = 3
        evolver.personality_vector.total_evolution_magnitude = 0.45
        evolver.personality_vector.last_evolution = time.time() - 3600
        
        status = evolver.get_personality_status()
        
        assert isinstance(status, dict)
        assert 'current_traits' in status
        assert 'personality_vector' in status
        assert 'evolution_metadata' in status
        assert 'configuration' in status
        
        # Check trait values
        traits = status['current_traits']
        assert len(traits) == 10
        for trait_name, value in traits.items():
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0
        
        # Check evolution metadata
        meta = status['evolution_metadata']
        assert meta['total_evolutions'] == 3
        assert meta['total_evolution_magnitude'] == 0.45
        assert meta['last_evolution'] > 0
    
    def test_persistence_state_save_load(self, evolver, temp_output_dir):
        """Test saving and loading personality state."""
        # Modify personality state
        evolver.personality_vector.curiosity = 0.8
        evolver.personality_vector.empathy = 0.3
        evolver.personality_vector.evolution_count = 2
        
        # Save state
        evolver._save_persistent_state()
        
        # Verify file was created
        assert evolver.state_file.exists()
        
        # Load state into new instance
        new_evolver = PersonalityEvolver.__new__(PersonalityEvolver)
        new_evolver.state_file = evolver.state_file
        new_evolver.history_file = evolver.history_file
        new_evolver.personality_vector = PersonalityVector()
        new_evolver._load_persistent_state()
        
        # Verify state was loaded correctly
        assert new_evolver.personality_vector.curiosity == 0.8
        assert new_evolver.personality_vector.empathy == 0.3
        assert new_evolver.personality_vector.evolution_count == 2
    
    def test_evolution_trace_persistence(self, evolver):
        """Test saving and loading evolution traces."""
        # Create test evolution trace
        trace = PersonalityEvolutionTrace(
            evolution_id="test_123",
            trigger_type="test",
            timestamp=time.time(),
            memory_analysis={},
            contradiction_analysis={},
            emotional_analysis={},
            dissonance_pressure=0.5,
            trait_changes=[],
            tone_changes={},
            mood_changes={},
            justification="Test evolution",
            natural_language_summary="Test summary",
            duration_ms=100.0,
            total_changes=2,
            evolution_magnitude=0.3
        )
        
        # Save trace
        evolver._save_evolution_trace(trace)
        
        # Verify file was created and contains data
        assert evolver.history_file.exists()
        
        # Load and verify
        history = evolver.get_evolution_history(limit=5)
        assert len(history) == 1
        assert history[0]['evolution_id'] == "test_123"
        assert history[0]['trigger_type'] == "test"
    
    def test_natural_language_summary_generation(self, evolver):
        """Test natural language summary generation."""
        # Create test trace with trait changes
        trait_changes = [
            PersonalityTraitChange(
                trait_name='empathy',
                old_value=0.5,
                new_value=0.7,
                change_magnitude=0.2,
                trigger_type='emotional_drift',
                trigger_details='Increased emotional language detected',
                confidence=0.8,
                timestamp=time.time()
            ),
            PersonalityTraitChange(
                trait_name='skepticism',
                old_value=0.6,
                new_value=0.8,
                change_magnitude=0.2,
                trigger_type='contradiction_stress',
                trigger_details='High contradiction pressure',
                confidence=0.9,
                timestamp=time.time()
            )
        ]
        
        trace = PersonalityEvolutionTrace(
            evolution_id="test",
            trigger_type="test",
            timestamp=time.time(),
            memory_analysis={},
            contradiction_analysis={},
            emotional_analysis={},
            dissonance_pressure=0.7,
            trait_changes=trait_changes,
            tone_changes={},
            mood_changes={},
            justification="Test",
            natural_language_summary="",
            duration_ms=100,
            total_changes=2,
            evolution_magnitude=0.4
        )
        
        summary = evolver.generate_natural_language_summary(trace)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert 'empathy' in summary.lower()
        assert 'skepticism' in summary.lower()
        assert 'increased' in summary.lower() or 'decreased' in summary.lower()


class TestCLIIntegration:
    """Test CLI command integration."""
    
    @pytest.fixture
    def mock_evolver(self):
        """Mock PersonalityEvolver for CLI testing."""
        evolver = Mock()
        evolver.get_personality_status.return_value = {
            'current_traits': {
                'curiosity': 0.7,
                'empathy': 0.6,
                'confidence': 0.5
            },
            'evolution_metadata': {
                'total_evolutions': 2,
                'last_evolution': time.time() - 3600,
                'total_evolution_magnitude': 0.3
            },
            'configuration': {
                'sensitivity_threshold': 0.3,
                'max_shift_rate': 0.2
            },
            'recent_evolution_history': [
                {
                    'summary': 'Increased empathy due to emotional language patterns'
                }
            ]
        }
        return evolver
    
    def test_personality_evolve_command(self, mock_evolver):
        """Test /personality_evolve CLI command."""
        from cortex.cli_commands import handle_personality_evolve_command
        
        # Mock evolution trace
        mock_trace = Mock()
        mock_trace.evolution_id = "evo_123"
        mock_trace.trigger_type = "manual"
        mock_trace.total_changes = 2
        mock_trace.evolution_magnitude = 0.25
        mock_trace.duration_ms = 150.0
        mock_trace.natural_language_summary = "Increased empathy and caution"
        mock_trace.trait_changes = [
            Mock(
                trait_name='empathy',
                old_value=0.5,
                new_value=0.7,
                trigger_details='Emotional language detected'
            )
        ]
        
        mock_evolver.evolve_personality.return_value = mock_trace
        
        with patch('cortex.cli_commands.get_personality_evolver', return_value=mock_evolver):
            result = handle_personality_evolve_command()
            
            assert result == 'continue'
            mock_evolver.evolve_personality.assert_called_once_with(
                trigger_type="manual", lookback_days=7
            )
    
    def test_personality_history_command(self, mock_evolver):
        """Test /personality_history CLI command."""
        from cortex.cli_commands import handle_personality_history_command
        
        # Mock evolution history
        mock_evolver.get_evolution_history.return_value = [
            {
                'evolution_id': 'evo_123',
                'timestamp': time.time() - 3600,
                'trigger_type': 'weekly_cadence',
                'total_changes': 2,
                'evolution_magnitude': 0.2,
                'natural_language_summary': 'Routine weekly adjustment',
                'trait_changes': []
            }
        ]
        
        with patch('cortex.cli_commands.get_personality_evolver', return_value=mock_evolver):
            result = handle_personality_history_command(limit=5)
            
            assert result == 'continue'
            mock_evolver.get_evolution_history.assert_called_once_with(5)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_memory_system(self, evolver):
        """Test graceful handling when memory system is unavailable."""
        evolver._memory_system = None
        
        analysis = evolver.analyze_memory_trends()
        
        # Should return default analysis structure without crashing
        assert isinstance(analysis, dict)
        assert 'memory_volume' in analysis
        assert analysis['memory_volume'] == 0
    
    def test_missing_dissonance_tracker(self, evolver):
        """Test graceful handling when dissonance tracker is unavailable."""
        evolver._dissonance_tracker = None
        
        analysis = evolver.analyze_contradiction_stress()
        
        # Should return default analysis structure
        assert isinstance(analysis, dict)
        assert 'pressure_level' in analysis
        assert analysis['pressure_level'] == 0.0
    
    def test_corrupted_state_file_recovery(self, evolver, temp_output_dir):
        """Test recovery from corrupted state file."""
        # Create corrupted state file
        with open(evolver.state_file, 'w') as f:
            f.write("invalid json content")
        
        # Should not crash when loading
        evolver._load_persistent_state()
        
        # Should still have valid default personality vector
        assert evolver.personality_vector.curiosity == 0.5
    
    def test_invalid_trait_name_handling(self, evolver):
        """Test handling of invalid trait names in changes."""
        invalid_change = PersonalityTraitChange(
            trait_name='nonexistent_trait',
            old_value=0.5,
            new_value=0.7,
            change_magnitude=0.2,
            trigger_type='test',
            trigger_details='test',
            confidence=1.0,
            timestamp=time.time()
        )
        
        # Should not crash and should not apply invalid changes
        changes_applied = evolver.apply_trait_changes([invalid_change])
        assert 'nonexistent_trait' not in changes_applied


class TestSingletonPattern:
    """Test the singleton pattern for PersonalityEvolver."""
    
    def test_singleton_instance(self):
        """Test that get_personality_evolver returns same instance."""
        with patch('agents.personality_evolver.get_config', return_value={}):
            # Clear singleton for clean test
            import agents.personality_evolver
            agents.personality_evolver._personality_evolver_instance = None
            
            evolver1 = get_personality_evolver()
            evolver2 = get_personality_evolver()
            
            assert evolver1 is evolver2
            assert isinstance(evolver1, PersonalityEvolver)


class TestConfigurableParameters:
    """Test that all parameters are configurable and not hardcoded [[memory:4199483]]."""
    
    def test_all_thresholds_configurable(self):
        """Test that all threshold values come from config."""
        custom_config = {
            'personality_evolution': {
                'sensitivity_threshold': 0.15,  # Non-default
                'max_shift_rate': 0.35,         # Non-default
                'trait_bounds': {'min': 0.1, 'max': 0.8},  # Non-default
                'weekly_cadence_hours': 72,     # Non-default
                'major_conflict_threshold': 0.6  # Non-default
            }
        }
        
        with patch('agents.personality_evolver.get_config', return_value=custom_config):
            evolver = PersonalityEvolver()
            
            # All values should match config, not hardcoded defaults
            assert evolver.sensitivity_threshold == 0.15
            assert evolver.max_shift_rate == 0.35
            assert evolver.trait_bounds == {'min': 0.1, 'max': 0.8}
            assert evolver.weekly_cadence_hours == 72
            assert evolver.major_conflict_threshold == 0.6
    
    def test_trait_evolution_rules_configurable(self, mock_config):
        """Test that trait evolution rules respect configuration bounds."""
        # Add trait-specific rules to config
        mock_config['personality_evolution']['trait_evolution'] = {
            'empathy': {
                'max_increase_per_cycle': 0.1,
                'max_decrease_per_cycle': 0.05
            },
            'skepticism': {
                'max_increase_per_cycle': 0.25,
                'max_decrease_per_cycle': 0.1
            }
        }
        
        with patch('agents.personality_evolver.get_config', return_value=mock_config):
            evolver = PersonalityEvolver()
            
            # Evolution should respect trait-specific limits
            # This would be tested in trait adjustment generation
            # Here we just verify the config is accessible
            trait_config = evolver.evolution_config.get('trait_evolution', {})
            assert 'empathy' in trait_config
            assert trait_config['empathy']['max_increase_per_cycle'] == 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])