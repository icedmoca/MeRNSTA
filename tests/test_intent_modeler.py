#!/usr/bin/env python3
"""
Test suite for agents/intent_modeler.py - Phase 7: Autonomous Intent Modeler
"""

import pytest
import tempfile
import os
import time
from unittest.mock import Mock, patch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.intent_modeler import (
    IntentPattern, EvolvedGoal, AutonomousIntentModeler, get_intent_modeler
)
from storage.drive_system import MotivationalDriveSystem
from agents.base import Goal


class TestIntentPattern:
    """Test IntentPattern functionality."""
    
    def test_intent_pattern_creation(self):
        """Test creating an IntentPattern."""
        pattern = IntentPattern(
            pattern_id="test_pattern_1",
            description="Strong interest in machine learning",
            confidence=0.85,
            supporting_evidence=["ML mentioned 5 times", "Seeking algorithms"],
            intent_category="exploration",
            temporal_frequency=0.3,
            context_triggers=["machine learning", "algorithms"]
        )
        
        assert pattern.pattern_id == "test_pattern_1"
        assert pattern.confidence == 0.85
        assert pattern.intent_category == "exploration"
        assert len(pattern.supporting_evidence) == 2
        assert len(pattern.context_triggers) == 2
        assert pattern.created_at > 0
        assert pattern.last_seen > 0
    
    def test_intent_pattern_serialization(self):
        """Test IntentPattern to/from dict conversion."""
        pattern = IntentPattern(
            pattern_id="test_pattern_2",
            description="Pattern for optimization intent",
            confidence=0.75,
            supporting_evidence=["Evidence 1"],
            intent_category="optimization",
            temporal_frequency=0.5,
            context_triggers=["optimization"]
        )
        
        # Convert to dict
        pattern_dict = pattern.to_dict()
        assert isinstance(pattern_dict, dict)
        assert pattern_dict["pattern_id"] == "test_pattern_2"
        assert pattern_dict["intent_category"] == "optimization"
        
        # Convert back from dict
        restored_pattern = IntentPattern.from_dict(pattern_dict)
        assert restored_pattern.pattern_id == pattern.pattern_id
        assert restored_pattern.confidence == pattern.confidence


class TestEvolvedGoal:
    """Test EvolvedGoal functionality."""
    
    def test_evolved_goal_creation(self):
        """Test creating an EvolvedGoal."""
        goal = EvolvedGoal(
            goal_id="evolved_goal_1",
            description="Evolved exploration goal",
            priority=0.8,
            strategy="deep_exploration",
            driving_motives={"curiosity": 0.9, "novelty": 0.6},
            tension_score=0.75,
            autonomy_level=0.9,
            source_intents=["pattern_1", "pattern_2"],
            abstraction_level=2,
            goal_lineage=["parent_goal_1"],
            adaptive_priority=0.85
        )
        
        assert goal.goal_id == "evolved_goal_1"
        assert goal.abstraction_level == 2
        assert len(goal.source_intents) == 2
        assert len(goal.goal_lineage) == 1
        assert goal.adaptive_priority == 0.85


class TestAutonomousIntentModeler:
    """Test AutonomousIntentModeler functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_intent_modeler.db")
        self.drive_system = MotivationalDriveSystem(self.db_path)
        self.intent_modeler = AutonomousIntentModeler(None, self.drive_system)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_intent_modeler_initialization(self):
        """Test AutonomousIntentModeler initialization."""
        assert self.intent_modeler.drive_system == self.drive_system
        assert self.intent_modeler.db_path == self.db_path
        assert self.intent_modeler.intent_confidence_threshold == 0.6
        assert self.intent_modeler.max_abstraction_levels == 4
        assert isinstance(self.intent_modeler.discovered_patterns, dict)
        assert isinstance(self.intent_modeler.evolved_goals, list)
    
    @patch('agents.intent_modeler.sqlite3.connect')
    def test_model_intent_from_patterns_no_data(self, mock_connect):
        """Test intent modeling with no data."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        result = self.intent_modeler.model_intent_from_patterns(["test tokens"], 24)
        
        assert "patterns" in result
        assert "analysis_timestamp" in result
        assert "facts_analyzed" in result
        assert result["facts_analyzed"] == 0
        assert len(result["patterns"]) == 0
    
    @patch('agents.intent_modeler.sqlite3.connect')
    def test_model_intent_from_patterns_with_data(self, mock_connect):
        """Test intent modeling with sample data."""
        # Mock facts data
        current_time = time.time()
        mock_facts = [
            (1, "machine learning", "is", "interesting", current_time, 0.8, 0.3, 123, "user1"),
            (2, "machine learning", "requires", "data", current_time - 100, 0.9, 0.2, 124, "user1"),
            (3, "algorithms", "solve", "problems", current_time - 200, 0.7, 0.5, 125, "user1"),
            (4, "uncertainty", "exists", "everywhere", current_time - 300, 0.4, 0.8, 126, "user1"),
        ]
        
        # Mock reflex patterns
        mock_reflex = [
            ("cycle1", "belief_clarification", True, "Clarify ML concepts", current_time - 50),
            ("cycle2", "exploration_goal", True, "Explore algorithms", current_time - 150),
            ("cycle3", "belief_clarification", False, "Failed clarification", current_time - 250),
        ]
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.side_effect = [mock_facts, mock_reflex]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        token_history = ["machine learning algorithms data science"]
        result = self.intent_modeler.model_intent_from_patterns(token_history, 24)
        
        assert "patterns" in result
        assert len(result["patterns"]) > 0
        assert result["facts_analyzed"] == 4
        
        # Should discover some patterns based on the data
        patterns = result["patterns"]
        pattern_categories = [p.intent_category for p in patterns]
        
        # Should include exploration pattern (ML mentioned multiple times)
        assert "exploration" in pattern_categories or "thematic_focus" in pattern_categories
    
    def test_evolve_goals_no_patterns(self):
        """Test goal evolution with no patterns."""
        base_intent = {"patterns": []}
        
        evolved_goals = self.intent_modeler.evolve_goals(base_intent)
        
        assert isinstance(evolved_goals, list)
        assert len(evolved_goals) == 0
    
    def test_evolve_goals_with_patterns(self):
        """Test goal evolution with intent patterns."""
        # Create mock patterns
        pattern1 = IntentPattern(
            pattern_id="pattern_exploration",
            description="Strong interest in exploring data science",
            confidence=0.8,
            supporting_evidence=["Multiple queries about ML"],
            intent_category="exploration",
            temporal_frequency=0.6,
            context_triggers=["data science", "machine learning"]
        )
        
        pattern2 = IntentPattern(
            pattern_id="pattern_optimization",
            description="Preference for efficient algorithms",
            confidence=0.75,
            supporting_evidence=["Success with optimization strategies"],
            intent_category="optimization",
            temporal_frequency=0.4,
            context_triggers=["optimization", "efficiency"]
        )
        
        base_intent = {
            "patterns": [pattern1, pattern2]
        }
        
        # Mock drive system to return dominant drives
        with patch.object(self.drive_system, 'get_current_dominant_drives') as mock_drives:
            mock_drives.return_value = {"curiosity": 0.8, "coherence": 0.6}
            
            evolved_goals = self.intent_modeler.evolve_goals(base_intent)
            
            assert isinstance(evolved_goals, list)
            assert len(evolved_goals) >= 2  # At least one goal per pattern
            
            # Check that goals have proper structure
            for goal in evolved_goals:
                assert isinstance(goal, EvolvedGoal)
                assert hasattr(goal, 'source_intents')
                assert hasattr(goal, 'abstraction_level')
                assert goal.autonomy_level >= 0.8  # Should be highly autonomous
    
    def test_adjust_priorities_empty_list(self):
        """Test priority adjustment with empty goal list."""
        adjusted_goals = self.intent_modeler.adjust_priorities([])
        assert adjusted_goals == []
    
    def test_adjust_priorities_with_goals(self):
        """Test priority adjustment with goals."""
        # Create test goals
        goal1 = Goal(
            goal_id="goal1",
            description="Test goal 1",
            priority=0.5,
            strategy="belief_clarification"
        )
        
        goal2 = EvolvedGoal(
            goal_id="goal2", 
            description="Test goal 2",
            priority=0.6,
            strategy="exploration_goal",
            driving_motives={"curiosity": 0.9},
            tension_score=0.7,
            autonomy_level=0.8,
            source_intents=[],
            abstraction_level=1,
            adaptive_priority=0.6
        )
        
        # Mock drive system to return dominant drives
        with patch.object(self.drive_system, 'get_current_dominant_drives') as mock_drives:
            mock_drives.return_value = {"curiosity": 0.9, "coherence": 0.4}
            
            adjusted_goals = self.intent_modeler.adjust_priorities([goal1, goal2])
            
            assert len(adjusted_goals) == 2
            
            # Goals should be sorted by priority (highest first)
            assert adjusted_goals[0].priority >= adjusted_goals[1].priority
            
            # Goal2 should get priority boost due to curiosity drive alignment
            goal2_adjusted = next(g for g in adjusted_goals if g.goal_id == "goal2")
            assert goal2_adjusted.adaptive_priority > 0.6  # Should be boosted
    
    def test_summarize_current_motives_no_drives(self):
        """Test motive summarization with no active drives."""
        with patch.object(self.drive_system, 'get_current_dominant_drives') as mock_drives:
            mock_drives.return_value = {}
            
            summary = self.intent_modeler.summarize_current_motives()
            
            assert isinstance(summary, str)
            assert "No active motivational drives detected" in summary
    
    def test_summarize_current_motives_with_drives(self):
        """Test motive summarization with active drives."""
        with patch.object(self.drive_system, 'get_current_dominant_drives') as mock_drives:
            mock_drives.return_value = {
                "curiosity": 0.85,
                "coherence": 0.6,
                "stability": 0.4
            }
            
            # Add some evolved goals to the modeler
            self.intent_modeler.evolved_goals = [
                EvolvedGoal(
                    goal_id="test_goal",
                    description="Test evolved goal",
                    priority=0.7,
                    strategy="exploration",
                    driving_motives={"curiosity": 0.8},
                    tension_score=0.6,
                    autonomy_level=0.9,
                    source_intents=[],
                    abstraction_level=2,
                    adaptive_priority=0.7,
                    created_at=time.time()
                )
            ]
            
            summary = self.intent_modeler.summarize_current_motives()
            
            assert isinstance(summary, str)
            assert "Current Motivational State" in summary
            assert "Curiosity" in summary  # Should mention highest drive
            assert "Recent Activity" in summary  # Should mention recent goals
    
    @patch('agents.intent_modeler.sqlite3.connect')
    def test_get_recent_patterns(self, mock_connect):
        """Test getting recent intent patterns."""
        # Mock database response
        current_time = time.time()
        mock_patterns = [
            ("pattern1", "Description 1", 0.8, '["evidence1"]', "exploration", 0.5, '["trigger1"]', current_time - 100, current_time - 50),
            ("pattern2", "Description 2", 0.7, '["evidence2"]', "optimization", 0.3, '["trigger2"]', current_time - 200, current_time - 100),
        ]
        
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = mock_patterns
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        patterns = self.intent_modeler._get_recent_patterns(24)
        
        assert len(patterns) == 2
        assert all(isinstance(p, IntentPattern) for p in patterns)
        assert patterns[0].pattern_id == "pattern1"
        assert patterns[0].confidence == 0.8


class TestIntentPatternAnalysis:
    """Test intent pattern analysis methods."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_analysis.db")
        self.drive_system = MotivationalDriveSystem(self.db_path)
        self.intent_modeler = AutonomousIntentModeler(None, self.drive_system)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_analyze_intent_patterns_subject_clustering(self):
        """Test subject clustering pattern detection."""
        facts = [
            (1, "python", "is", "language", time.time(), 0.8, 0.2, 123, "user1"),
            (2, "python", "has", "libraries", time.time(), 0.9, 0.1, 124, "user1"),
            (3, "python", "enables", "ML", time.time(), 0.7, 0.3, 125, "user1"),
            (4, "java", "is", "verbose", time.time(), 0.6, 0.4, 126, "user1"),
        ]
        
        reflex_patterns = []
        token_history = ["python programming"]
        
        analysis = self.intent_modeler._analyze_intent_patterns(facts, reflex_patterns, token_history)
        
        assert "patterns" in analysis
        patterns = analysis["patterns"]
        
        # Should detect exploration pattern for "python"
        exploration_patterns = [p for p in patterns if p.intent_category == "exploration"]
        assert len(exploration_patterns) > 0
        
        python_pattern = next((p for p in exploration_patterns if "python" in p.description.lower()), None)
        assert python_pattern is not None
        assert python_pattern.confidence > 0.0
    
    def test_analyze_intent_patterns_strategy_preferences(self):
        """Test strategy preference pattern detection."""
        facts = []
        reflex_patterns = [
            ("cycle1", "belief_clarification", True, "Goal 1", time.time()),
            ("cycle2", "belief_clarification", True, "Goal 2", time.time()),
            ("cycle3", "belief_clarification", True, "Goal 3", time.time()),
            ("cycle4", "cluster_reassessment", False, "Goal 4", time.time()),
        ]
        token_history = []
        
        analysis = self.intent_modeler._analyze_intent_patterns(facts, reflex_patterns, token_history)
        
        patterns = analysis["patterns"]
        optimization_patterns = [p for p in patterns if p.intent_category == "optimization"]
        
        # Should detect preference for belief_clarification strategy
        assert len(optimization_patterns) > 0
        belief_pattern = next((p for p in optimization_patterns if "belief_clarification" in p.description), None)
        assert belief_pattern is not None
        assert belief_pattern.confidence >= 0.7  # High success rate
    
    def test_analyze_intent_patterns_volatility_response(self):
        """Test volatility response pattern detection."""
        facts = [
            (1, "topic1", "is", "unclear", time.time(), 0.4, 0.9, 123, "user1"),  # High volatility
            (2, "topic2", "seems", "confusing", time.time(), 0.3, 0.8, 124, "user1"),  # High volatility
            (3, "topic3", "requires", "study", time.time(), 0.5, 0.9, 125, "user1"),  # High volatility
            (4, "topic4", "is", "clear", time.time(), 0.9, 0.1, 126, "user1"),  # Low volatility
        ]
        
        reflex_patterns = []
        token_history = []
        
        analysis = self.intent_modeler._analyze_intent_patterns(facts, reflex_patterns, token_history)
        
        patterns = analysis["patterns"]
        stabilization_patterns = [p for p in patterns if p.intent_category == "stabilization"]
        
        # Should detect stabilization intent
        assert len(stabilization_patterns) > 0
        stab_pattern = stabilization_patterns[0]
        assert "stabilize" in stab_pattern.description.lower() or "volatility" in stab_pattern.description.lower()


class TestGoalEvolution:
    """Test goal evolution functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_evolution.db")
        self.drive_system = MotivationalDriveSystem(self.db_path)
        self.intent_modeler = AutonomousIntentModeler(None, self.drive_system)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_evolve_goal_from_pattern(self):
        """Test evolving a single goal from a pattern."""
        pattern = IntentPattern(
            pattern_id="test_pattern",
            description="Strong curiosity about AI",
            confidence=0.8,
            supporting_evidence=["Multiple AI queries"],
            intent_category="curiosity",
            temporal_frequency=0.7,
            context_triggers=["AI", "neural networks"]
        )
        
        drives = {"curiosity": 0.9, "coherence": 0.5}
        
        goal = self.intent_modeler._evolve_goal_from_pattern(pattern, drives)
        
        assert goal is not None
        assert isinstance(goal, EvolvedGoal)
        assert goal.strategy == "knowledge_acquisition"  # Curiosity maps to knowledge acquisition
        assert goal.source_intents == ["test_pattern"]
        assert goal.abstraction_level == 2
        assert goal.autonomy_level == 0.9
        assert goal.adaptive_priority > pattern.confidence  # Should get drive boost
    
    def test_create_meta_goals(self):
        """Test creating meta-goals from multiple patterns."""
        patterns = [
            IntentPattern(
                pattern_id="exploration1",
                description="Exploration pattern 1",
                confidence=0.7,
                supporting_evidence=[],
                intent_category="exploration",
                temporal_frequency=0.5,
                context_triggers=[]
            ),
            IntentPattern(
                pattern_id="exploration2", 
                description="Exploration pattern 2",
                confidence=0.8,
                supporting_evidence=[],
                intent_category="exploration",
                temporal_frequency=0.6,
                context_triggers=[]
            ),
            IntentPattern(
                pattern_id="optimization1",
                description="Single optimization pattern",
                confidence=0.6,
                supporting_evidence=[],
                intent_category="optimization",
                temporal_frequency=0.3,
                context_triggers=[]
            )
        ]
        
        drives = {"curiosity": 0.8, "coherence": 0.7}
        
        meta_goals = self.intent_modeler._create_meta_goals(patterns, drives)
        
        # Should create meta-goal for exploration (2 patterns) but not optimization (1 pattern)
        assert len(meta_goals) == 1
        meta_goal = meta_goals[0]
        assert "exploration" in meta_goal.goal_id
        assert len(meta_goal.source_intents) == 2
        assert meta_goal.abstraction_level == 3  # Meta-goals are level 3
        assert meta_goal.autonomy_level == 0.95  # Very high autonomy


class TestGlobalIntentModeler:
    """Test global intent modeler functionality."""
    
    def test_get_intent_modeler_singleton(self):
        """Test that get_intent_modeler returns consistent instance."""
        # Create temporary database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_singleton.db")
        drive_system = MotivationalDriveSystem(db_path)
        
        try:
            # Clear any existing instance
            if hasattr(get_intent_modeler, '_instance'):
                delattr(get_intent_modeler, '_instance')
            
            # Get two instances
            modeler1 = get_intent_modeler(None, drive_system)
            modeler2 = get_intent_modeler(None, drive_system)
            
            # Should be the same instance
            assert modeler1 is modeler2
            
        finally:
            # Cleanup
            if hasattr(get_intent_modeler, '_instance'):
                delattr(get_intent_modeler, '_instance')
            if os.path.exists(db_path):
                os.remove(db_path)
            os.rmdir(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])