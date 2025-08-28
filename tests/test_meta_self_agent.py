#!/usr/bin/env python3
"""
Comprehensive tests for the MetaSelfAgent Phase 27 pt 2 implementation.

Tests cover:
- Health monitoring functionality
- Goal scaffolding system
- Meta-cognitive analysis
- CLI command integration
- Persistence and logging
- Configuration handling
"""

import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Test imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.meta_self_agent import (
    MetaSelfAgent, 
    CognitiveHealthMetrics, 
    MetaGoal, 
    get_meta_self_agent
)


class TestMetaSelfAgent(unittest.TestCase):
    """Test suite for MetaSelfAgent core functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use temporary directory for test logs
        self.temp_dir = tempfile.mkdtemp()
        self.test_log_path = Path(self.temp_dir) / "test_meta_self_log.jsonl"
        
        # Mock configuration
        self.mock_config = {
            'meta_self_agent': {
                'health_thresholds': {
                    'memory_bloat_threshold': 0.3,
                    'agent_performance_threshold': 0.6,
                    'dissonance_threshold': 0.7,
                    'contract_drift_threshold': 0.4,
                    'critical_health_threshold': 0.4
                },
                'check_interval_minutes': 30,
                'deep_analysis_interval_hours': 6,
                'max_active_meta_goals': 5,
                'goal_priority_weights': {
                    'critical_system_health': 0.9,
                    'performance_decline': 0.8,
                    'memory_optimization': 0.6,
                    'dissonance_resolution': 0.7,
                    'preventive_maintenance': 0.4
                }
            }
        }
        
        # Create MetaSelfAgent instance with mocked config
        with patch('agents.meta_self_agent.get_config', return_value=self.mock_config):
            with patch.object(MetaSelfAgent, '_load_persistent_state'):
                self.agent = MetaSelfAgent()
                self.agent.log_path = self.test_log_path
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test MetaSelfAgent initialization."""
        self.assertEqual(self.agent.name, "meta_self")
        self.assertIsNotNone(self.agent.config)
        self.assertEqual(self.agent.memory_bloat_threshold, 0.3)
        self.assertEqual(self.agent.max_active_meta_goals, 5)
        self.assertIsInstance(self.agent.generated_goals, dict)
        self.assertIsInstance(self.agent.health_history, type(self.agent.health_history))
    
    def test_get_agent_instructions(self):
        """Test agent instruction generation."""
        instructions = self.agent.get_agent_instructions()
        self.assertIn("MetaSelfAgent", instructions)
        self.assertIn("cognitive system health", instructions)
        self.assertIn("meta-goals", instructions)
    
    def test_cognitive_health_metrics_creation(self):
        """Test CognitiveHealthMetrics dataclass creation."""
        timestamp = time.time()
        metrics = CognitiveHealthMetrics(
            timestamp=timestamp,
            overall_health_score=0.8,
            memory_health=0.7,
            agent_performance_health=0.9,
            dissonance_health=0.8,
            contract_alignment_health=0.6,
            reflection_quality_health=0.7,
            total_memory_facts=1000,
            memory_bloat_ratio=0.2,
            average_agent_performance=0.75,
            active_dissonance_regions=2,
            unresolved_contradictions=1,
            contract_drift_count=0,
            anomalies_detected=["test anomaly"],
            critical_issues=[],
            recommendations=["test recommendation"]
        )
        
        self.assertEqual(metrics.timestamp, timestamp)
        self.assertEqual(metrics.overall_health_score, 0.8)
        self.assertEqual(len(metrics.anomalies_detected), 1)
        self.assertEqual(len(metrics.recommendations), 1)
    
    def test_meta_goal_creation(self):
        """Test MetaGoal dataclass creation and conversion."""
        current_time = time.time()
        meta_goal = MetaGoal(
            goal_id="test_goal_1",
            goal_type="memory_optimization",
            description="Test memory optimization goal",
            priority=0.8,
            urgency=0.6,
            justification="Test justification",
            target_component="memory_system",
            expected_outcome="Improved memory efficiency",
            created_at=current_time
        )
        
        self.assertEqual(meta_goal.goal_id, "test_goal_1")
        self.assertEqual(meta_goal.priority, 0.8)
        self.assertEqual(meta_goal.created_at, current_time)
        
        # Test conversion to Goal object
        goal = meta_goal.to_goal()
        self.assertEqual(goal.goal_id, "meta_test_goal_1")
        self.assertEqual(goal.description, "Test memory optimization goal")
        self.assertEqual(goal.strategy, "meta_cognitive")
        self.assertIn("meta_type", goal.metadata)
    
    @patch('agents.meta_self_agent.statistics.mean')
    def test_memory_health_analysis_healthy(self, mock_mean):
        """Test memory health analysis for healthy system."""
        # Mock memory system
        mock_memory_system = Mock()
        mock_facts = [Mock(confidence=0.8), Mock(confidence=0.9), Mock(confidence=0.7)]
        mock_memory_system.get_all_facts.return_value = mock_facts
        
        self.agent._memory_system = mock_memory_system
        
        result = self.agent._analyze_memory_health()
        
        self.assertGreater(result['score'], 0.5)
        self.assertEqual(result['total_facts'], 3)
        self.assertLess(result['bloat_ratio'], 0.3)
        self.assertEqual(len(result['critical_issues']), 0)
    
    @patch('agents.meta_self_agent.statistics.mean')
    def test_memory_health_analysis_bloated(self, mock_mean):
        """Test memory health analysis for bloated system."""
        # Mock memory system with low confidence facts
        mock_memory_system = Mock()
        mock_facts = [Mock(confidence=0.1) for _ in range(10)]  # Many low confidence facts
        mock_memory_system.get_all_facts.return_value = mock_facts
        
        self.agent._memory_system = mock_memory_system
        
        result = self.agent._analyze_memory_health()
        
        self.assertLess(result['score'], 0.8)  # Should be reduced due to bloat
        self.assertGreater(result['bloat_ratio'], 0.5)  # High bloat ratio
        self.assertGreater(len(result['anomalies']), 0)  # Should detect bloat
    
    @patch('agents.meta_self_agent.statistics.mean')
    def test_agent_performance_analysis(self, mock_mean):
        """Test agent performance analysis."""
        # Mock agent registry
        mock_registry = Mock()
        mock_agent1 = Mock()
        mock_agent1.get_lifecycle_metrics.return_value = {'current_performance': 0.8}
        mock_agent2 = Mock()
        mock_agent2.get_lifecycle_metrics.return_value = {'current_performance': 0.4}  # Underperforming
        
        mock_registry.get_all_agents.return_value = {
            'agent1': mock_agent1,
            'agent2': mock_agent2
        }
        
        self.agent._agent_registry = mock_registry
        mock_mean.return_value = 0.6
        
        result = self.agent._analyze_agent_performance()
        
        self.assertEqual(result['average_performance'], 0.6)
        self.assertGreater(len(result['anomalies']), 0)  # Should detect underperforming agent
    
    def test_dissonance_health_analysis(self):
        """Test dissonance health analysis."""
        # Mock dissonance tracker
        mock_tracker = Mock()
        mock_region1 = Mock(pressure_score=0.8, duration=3600)
        mock_region2 = Mock(pressure_score=0.9, duration=86400 + 1)  # > 24 hours
        
        mock_tracker.dissonance_regions = {
            'region1': mock_region1,
            'region2': mock_region2
        }
        
        self.agent._dissonance_tracker = mock_tracker
        
        result = self.agent._analyze_dissonance_health()
        
        self.assertEqual(result['active_regions'], 2)
        self.assertEqual(result['unresolved_contradictions'], 1)  # region2 is unresolved
        self.assertGreater(len(result['anomalies']), 0)  # High dissonance should be detected
    
    def test_contract_alignment_analysis(self):
        """Test contract alignment analysis."""
        # Mock agent registry with contract issues
        mock_registry = Mock()
        mock_agent1 = Mock()
        mock_agent1.contract = Mock(last_updated=datetime.now() - timedelta(days=35))  # Stale
        mock_agent1.get_lifecycle_metrics.return_value = {'contract_alignment': 0.3}  # Drifted
        
        mock_agent2 = Mock()
        mock_agent2.contract = None  # Missing contract
        
        mock_registry.get_all_agents.return_value = {
            'agent1': mock_agent1,
            'agent2': mock_agent2
        }
        
        self.agent._agent_registry = mock_registry
        
        result = self.agent._analyze_contract_alignment()
        
        self.assertEqual(result['drift_count'], 2)  # Both agents have issues
        self.assertLess(result['score'], 0.8)  # Score should be reduced
        self.assertGreater(len(result['anomalies']), 0)  # Should detect issues
    
    def test_health_check_integration(self):
        """Test complete health check integration."""
        # Mock all component analysis methods
        with patch.object(self.agent, '_analyze_memory_health') as mock_memory:
            with patch.object(self.agent, '_analyze_agent_performance') as mock_agent:
                with patch.object(self.agent, '_analyze_dissonance_health') as mock_dissonance:
                    with patch.object(self.agent, '_analyze_contract_alignment') as mock_contract:
                        with patch.object(self.agent, '_analyze_reflection_quality') as mock_reflection:
                            
                            # Set up mock returns
                            mock_memory.return_value = {'score': 0.8, 'anomalies': [], 'critical_issues': [], 'recommendations': []}
                            mock_agent.return_value = {'score': 0.7, 'anomalies': [], 'critical_issues': [], 'recommendations': []}
                            mock_dissonance.return_value = {'score': 0.9, 'anomalies': [], 'critical_issues': [], 'recommendations': []}
                            mock_contract.return_value = {'score': 0.6, 'anomalies': [], 'critical_issues': [], 'recommendations': []}
                            mock_reflection.return_value = {'score': 0.8, 'anomalies': [], 'critical_issues': [], 'recommendations': []}
                            
                            health_metrics = self.agent.perform_health_check()
                            
                            self.assertIsInstance(health_metrics, CognitiveHealthMetrics)
                            self.assertAlmostEqual(health_metrics.overall_health_score, 0.76, places=1)  # Average of scores
                            self.assertEqual(len(self.agent.health_history), 1)
                            self.assertIsNotNone(self.agent.last_health_check)
    
    def test_goal_generation_for_critical_issues(self):
        """Test goal generation when critical issues are detected."""
        # Create health metrics with critical issues
        health_metrics = CognitiveHealthMetrics(
            timestamp=time.time(),
            overall_health_score=0.3,  # Below critical threshold
            memory_health=0.5,
            agent_performance_health=0.5,
            dissonance_health=0.5,
            contract_alignment_health=0.5,
            reflection_quality_health=0.5,
            total_memory_facts=1000,
            memory_bloat_ratio=0.6,
            average_agent_performance=0.5,
            active_dissonance_regions=0,
            unresolved_contradictions=0,
            contract_drift_count=0,
            anomalies_detected=[],
            critical_issues=["Critical memory bloat detected"],
            recommendations=[]
        )
        
        # Mock task selector to avoid actual queuing
        with patch.object(self.agent, '_queue_meta_goal'):
            self.agent._generate_improvement_goals(health_metrics)
            
            # Should generate goals for critical issues
            self.assertGreater(len(self.agent.generated_goals), 0)
            
            # Check that a memory cleanup goal was generated
            memory_goals = [g for g in self.agent.generated_goals.values() 
                          if g.goal_type == "memory_optimization"]
            self.assertGreater(len(memory_goals), 0)
    
    def test_goal_effectiveness_analysis(self):
        """Test analysis of goal effectiveness."""
        # Add some test goals with different statuses
        current_time = time.time()
        
        completed_goal = MetaGoal(
            goal_id="completed_1",
            goal_type="memory_optimization",
            description="Completed goal",
            priority=0.8,
            urgency=0.7,
            justification="Test",
            target_component="memory",
            expected_outcome="Improvement",
            created_at=current_time - 3600,
            completed_at=current_time - 1800
        )
        
        pending_goal = MetaGoal(
            goal_id="pending_1",
            goal_type="agent_adjustment",
            description="Pending goal",
            priority=0.6,
            urgency=0.5,
            justification="Test",
            target_component="agents",
            expected_outcome="Improvement",
            created_at=current_time - 1800
        )
        
        self.agent.generated_goals = {
            "completed_1": completed_goal,
            "pending_1": pending_goal
        }
        
        effectiveness = self.agent._analyze_goal_effectiveness()
        
        self.assertEqual(effectiveness['total_generated'], 2)
        self.assertEqual(effectiveness['completed'], 1)
        self.assertEqual(effectiveness['pending'], 1)
    
    def test_health_trends_analysis(self):
        """Test analysis of health trends over time."""
        # Add multiple health metrics to history
        timestamps = [time.time() - 3600 * i for i in range(5, 0, -1)]
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]  # Improving trend
        
        for timestamp, score in zip(timestamps, scores):
            metrics = CognitiveHealthMetrics(
                timestamp=timestamp,
                overall_health_score=score,
                memory_health=score,
                agent_performance_health=score,
                dissonance_health=score,
                contract_alignment_health=score,
                reflection_quality_health=score,
                total_memory_facts=1000,
                memory_bloat_ratio=0.1,
                average_agent_performance=score,
                active_dissonance_regions=0,
                unresolved_contradictions=0,
                contract_drift_count=0,
                anomalies_detected=[],
                critical_issues=[],
                recommendations=[]
            )
            self.agent.health_history.append(metrics)
        
        trends = self.agent._analyze_health_trends()
        
        self.assertIn('overall_health_score', trends)
        self.assertEqual(trends['overall_health_score']['direction'], 'improving')
        self.assertGreater(trends['overall_health_score']['trend'], 0)
    
    def test_anomaly_pattern_detection(self):
        """Test detection of recurring anomaly patterns."""
        # Add health metrics with recurring anomalies
        for i in range(5):
            metrics = CognitiveHealthMetrics(
                timestamp=time.time() - 3600 * i,
                overall_health_score=0.8,
                memory_health=0.8,
                agent_performance_health=0.8,
                dissonance_health=0.8,
                contract_alignment_health=0.8,
                reflection_quality_health=0.8,
                total_memory_facts=1000,
                memory_bloat_ratio=0.1,
                average_agent_performance=0.8,
                active_dissonance_regions=0,
                unresolved_contradictions=0,
                contract_drift_count=0,
                anomalies_detected=["High memory bloat", "Agent performance decline"],
                critical_issues=[],
                recommendations=[]
            )
            self.agent.health_history.append(metrics)
        
        patterns = self.agent._detect_anomaly_patterns()
        
        self.assertIn('frequent_anomalies', patterns)
        self.assertGreater(len(patterns['frequent_anomalies']), 0)
        # "high memory bloat" should appear 5 times
        self.assertIn('high memory bloat', patterns['frequent_anomalies'])
    
    def test_strategic_recommendations(self):
        """Test generation of strategic recommendations."""
        # Set up health history with declining trend
        declining_scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        for i, score in enumerate(declining_scores):
            metrics = CognitiveHealthMetrics(
                timestamp=time.time() - 3600 * (len(declining_scores) - i),
                overall_health_score=score,
                memory_health=score,
                agent_performance_health=score,
                dissonance_health=score,
                contract_alignment_health=score,
                reflection_quality_health=score,
                total_memory_facts=1000,
                memory_bloat_ratio=0.5 if score < 0.5 else 0.1,
                average_agent_performance=score,
                active_dissonance_regions=5 if score < 0.6 else 1,
                unresolved_contradictions=0,
                contract_drift_count=3 if score < 0.7 else 0,
                anomalies_detected=[],
                critical_issues=[],
                recommendations=[]
            )
            self.agent.health_history.append(metrics)
        
        recommendations = self.agent._generate_strategic_recommendations()
        
        self.assertGreater(len(recommendations), 0)
        # Should recommend emergency protocol for critical health
        critical_recommendations = [r for r in recommendations if "CRITICAL" in r]
        self.assertGreater(len(critical_recommendations), 0)
    
    def test_timing_methods(self):
        """Test timing control methods."""
        # Test health check timing
        self.assertTrue(self.agent.should_run_health_check())  # No previous check
        
        self.agent.last_health_check = datetime.now()
        self.assertFalse(self.agent.should_run_health_check())  # Recent check
        
        # Test deep analysis timing
        self.assertTrue(self.agent.should_run_deep_analysis())  # No previous analysis
        
        self.agent.last_deep_analysis = datetime.now()
        self.assertFalse(self.agent.should_run_deep_analysis())  # Recent analysis
    
    def test_goal_completion_tracking(self):
        """Test goal completion marking and tracking."""
        # Add a test goal
        goal = MetaGoal(
            goal_id="test_completion",
            goal_type="memory_optimization",
            description="Test goal",
            priority=0.8,
            urgency=0.7,
            justification="Test",
            target_component="memory",
            expected_outcome="Improvement",
            created_at=time.time()
        )
        
        self.agent.generated_goals["test_completion"] = goal
        
        # Mark as completed
        success = self.agent.mark_goal_completed("test_completion", "Successfully optimized memory")
        
        self.assertTrue(success)
        self.assertIsNotNone(goal.completed_at)
        self.assertTrue(goal.outcome_verified)
    
    def test_response_generation(self):
        """Test agent response generation for different message types."""
        # Mock health check
        with patch.object(self.agent, 'perform_health_check') as mock_health:
            mock_metrics = Mock()
            mock_metrics.overall_health_score = 0.8
            mock_metrics.critical_issues = []
            mock_metrics.anomalies_detected = ["test anomaly"]
            mock_health.return_value = mock_metrics
            
            # Test health-related response
            response = self.agent.respond("system health status")
            self.assertIn("health", response.lower())
            
            # Test goals-related response
            response = self.agent.respond("show goals")
            self.assertIn("goal", response.lower())
            
            # Test reflection response
            response = self.agent.respond("analyze system")
            self.assertIn("analysis", response.lower())
    
    def test_persistent_state_management(self):
        """Test saving and loading persistent state."""
        # Add test goals
        goal = MetaGoal(
            goal_id="persistent_test",
            goal_type="memory_optimization",
            description="Test persistence",
            priority=0.8,
            urgency=0.7,
            justification="Test",
            target_component="memory",
            expected_outcome="Improvement",
            created_at=time.time()
        )
        
        self.agent.generated_goals["persistent_test"] = goal
        
        # Save state
        self.agent._save_persistent_state()
        
        # Verify log file was created and contains data
        self.assertTrue(self.test_log_path.exists())
        
        with open(self.test_log_path, 'r') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)
            
            # Check that state checkpoint was logged
            checkpoint_found = False
            for line in lines:
                entry = json.loads(line)
                if entry.get('type') == 'state_checkpoint':
                    checkpoint_found = True
                    self.assertEqual(entry['data']['total_goals'], 1)
                    break
            
            self.assertTrue(checkpoint_found)


class TestMetaSelfAgentCLIIntegration(unittest.TestCase):
    """Test CLI command integration."""
    
    def setUp(self):
        """Set up CLI test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_log_path = Path(self.temp_dir) / "test_cli_log.jsonl"
        
        # Mock configuration
        self.mock_config = {
            'meta_self_agent': {
                'health_thresholds': {'memory_bloat_threshold': 0.3},
                'check_interval_minutes': 30,
                'max_active_meta_goals': 5
            }
        }
    
    def tearDown(self):
        """Clean up CLI test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('cortex.cli_commands.get_meta_self_agent')
    def test_meta_status_command(self, mock_get_agent):
        """Test /meta_status CLI command."""
        from cortex.cli_commands import handle_meta_status_command
        
        # Mock agent and health metrics
        mock_agent = Mock()
        mock_metrics = Mock()
        mock_metrics.overall_health_score = 0.8
        mock_metrics.critical_issues = []
        mock_agent.perform_health_check.return_value = mock_metrics
        mock_agent._generate_health_report.return_value = "Test health report"
        
        mock_get_agent.return_value = mock_agent
        
        # Execute command
        result = handle_meta_status_command()
        
        self.assertEqual(result, 'continue')
        mock_agent.perform_health_check.assert_called_once()
        mock_agent._generate_health_report.assert_called_once()
    
    @patch('cortex.cli_commands.get_meta_self_agent')
    def test_meta_reflect_command(self, mock_get_agent):
        """Test /meta_reflect CLI command."""
        from cortex.cli_commands import handle_meta_reflect_command
        
        # Mock agent
        mock_agent = Mock()
        mock_agent._generate_introspective_analysis.return_value = "Test introspection report"
        
        mock_get_agent.return_value = mock_agent
        
        # Execute command
        result = handle_meta_reflect_command()
        
        self.assertEqual(result, 'continue')
        mock_agent._generate_introspective_analysis.assert_called_once()
    
    @patch('cortex.cli_commands.get_meta_self_agent')
    def test_meta_goals_command(self, mock_get_agent):
        """Test /meta_goals CLI command."""
        from cortex.cli_commands import handle_meta_goals_command
        
        # Mock agent with goals
        mock_agent = Mock()
        mock_goal = Mock()
        mock_goal.description = "Test goal"
        mock_goal.goal_type = "memory_optimization"
        mock_goal.priority = 0.8
        mock_goal.executed_at = None
        mock_goal.completed_at = None
        
        mock_agent.get_recent_meta_goals.return_value = [mock_goal]
        mock_agent.generated_goals = {"test": mock_goal}
        
        mock_get_agent.return_value = mock_agent
        
        # Execute command
        result = handle_meta_goals_command(5)
        
        self.assertEqual(result, 'continue')
        mock_agent.get_recent_meta_goals.assert_called_once_with(5)


class TestMetaSelfAgentTaskIntegration(unittest.TestCase):
    """Test Celery task integration."""
    
    @patch('tasks.task_queue.get_meta_self_agent')
    def test_health_check_task(self, mock_get_agent):
        """Test meta-self health check task."""
        from tasks.task_queue import meta_self_health_check_task
        
        # Mock agent and health check
        mock_agent = Mock()
        mock_agent.should_run_health_check.return_value = True
        
        mock_metrics = Mock()
        mock_metrics.overall_health_score = 0.8
        mock_metrics.critical_issues = []
        mock_metrics.anomalies_detected = ["test anomaly"]
        
        mock_agent.perform_health_check.return_value = mock_metrics
        mock_agent.generated_goals = {"test": Mock()}
        
        mock_get_agent.return_value = mock_agent
        
        # Mock Celery task context
        mock_task = Mock()
        mock_task.request.retries = 0
        mock_task.max_retries = 2
        
        # Execute task
        result = meta_self_health_check_task(mock_task)
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['overall_health_score'], 0.8)
        self.assertEqual(result['critical_issues'], 0)
        self.assertEqual(result['anomalies_detected'], 1)
        self.assertEqual(result['goals_generated'], 1)
    
    @patch('tasks.task_queue.get_meta_self_agent')
    def test_deep_analysis_task(self, mock_get_agent):
        """Test meta-self deep analysis task."""
        from tasks.task_queue import meta_self_deep_analysis_task
        
        # Mock agent and analysis
        mock_agent = Mock()
        mock_agent.should_run_deep_analysis.return_value = True
        
        mock_analysis = {
            'timestamp': time.time(),
            'health_metrics': {'overall_health_score': 0.75},
            'pattern_detection': {'pattern1': 'data'},
            'recommendations': ['rec1', 'rec2'],
            'system_insights': ['insight1']
        }
        
        mock_agent.trigger_introspective_analysis.return_value = mock_analysis
        
        mock_get_agent.return_value = mock_agent
        
        # Mock Celery task context
        mock_task = Mock()
        mock_task.request.retries = 0
        mock_task.max_retries = 1
        
        # Execute task
        result = meta_self_deep_analysis_task(mock_task)
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['health_score'], 0.75)
        self.assertEqual(result['patterns_detected'], 1)
        self.assertEqual(result['recommendations'], 2)
        self.assertEqual(result['insights'], 1)
    
    @patch('tasks.task_queue.get_meta_self_agent')
    def test_task_skip_when_not_due(self, mock_get_agent):
        """Test task skipping when not due to run."""
        from tasks.task_queue import meta_self_health_check_task
        
        # Mock agent that doesn't need health check
        mock_agent = Mock()
        mock_agent.should_run_health_check.return_value = False
        
        mock_get_agent.return_value = mock_agent
        
        # Mock Celery task context
        mock_task = Mock()
        
        # Execute task
        result = meta_self_health_check_task(mock_task)
        
        self.assertEqual(result['status'], 'skipped')
        self.assertEqual(result['reason'], 'Health check not due yet')


class TestMetaSelfAgentConfigIntegration(unittest.TestCase):
    """Test configuration integration."""
    
    def test_config_loading(self):
        """Test that configuration is properly loaded and applied."""
        test_config = {
            'meta_self_agent': {
                'health_thresholds': {
                    'memory_bloat_threshold': 0.5,
                    'agent_performance_threshold': 0.7,
                    'critical_health_threshold': 0.3
                },
                'check_interval_minutes': 45,
                'max_active_meta_goals': 8,
                'goal_priority_weights': {
                    'critical_system_health': 0.95
                }
            }
        }
        
        with patch('agents.meta_self_agent.get_config', return_value=test_config):
            with patch.object(MetaSelfAgent, '_load_persistent_state'):
                agent = MetaSelfAgent()
                
                # Verify configuration was applied
                self.assertEqual(agent.memory_bloat_threshold, 0.5)
                self.assertEqual(agent.agent_performance_threshold, 0.7)
                self.assertEqual(agent.critical_health_threshold, 0.3)
                self.assertEqual(agent.check_interval, 45 * 60)
                self.assertEqual(agent.max_active_meta_goals, 8)
                self.assertEqual(agent.goal_priority_weights['critical_system_health'], 0.95)
    
    def test_default_config_fallback(self):
        """Test fallback to default values when config is missing."""
        empty_config = {}
        
        with patch('agents.meta_self_agent.get_config', return_value=empty_config):
            with patch.object(MetaSelfAgent, '_load_persistent_state'):
                agent = MetaSelfAgent()
                
                # Should use default values
                self.assertIsNotNone(agent.memory_bloat_threshold)
                self.assertIsNotNone(agent.check_interval)
                self.assertIsNotNone(agent.max_active_meta_goals)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)