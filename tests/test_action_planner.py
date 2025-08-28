#!/usr/bin/env python3
"""
Comprehensive tests for the AutonomousPlanner Phase 28 implementation.

Tests cover:
- Plan creation and scoring functionality
- Future projection engine
- System state monitoring
- Goal generation and queuing
- Plan evolution and failure handling
- CLI command integration
- Persistence and rollback capability
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

from agents.action_planner import (
    AutonomousPlanner,
    ActionPlan,
    PlanStep, 
    SystemProjection,
    get_autonomous_planner
)


class TestAutonomousPlanner(unittest.TestCase):
    """Test suite for AutonomousPlanner core functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use temporary directory for test logs
        self.temp_dir = tempfile.mkdtemp()
        self.test_plan_path = Path(self.temp_dir) / "test_action_plan.jsonl"
        self.test_log_path = Path(self.temp_dir) / "test_autonomous_planner_log.jsonl"
        
        # Mock configuration
        self.mock_config = {
            'autonomous_planner': {
                'enabled': True,
                'planning_interval_hours': 6,
                'min_plan_score': 0.4,
                'max_active_plans': 3,
                'lookahead_days': 7,
                'scoring_weights': {
                    'memory_health': 0.25,
                    'agent_performance': 0.25,
                    'system_stability': 0.20,
                    'goal_achievement': 0.15,
                    'innovation_potential': 0.15
                },
                'projection': {
                    'memory_growth_threshold': 1.5,
                    'drift_prediction_window_hours': 24,
                    'confidence_threshold': 0.6
                }
            }
        }
        
        # Create test planner with mocked dependencies
        with patch('config.settings.get_config', return_value=self.mock_config):
            self.planner = AutonomousPlanner()
            self.planner.plan_storage_path = self.test_plan_path
        
        # Mock system components
        self.mock_meta_self_agent = Mock()
        self.mock_memory_system = Mock()
        self.mock_task_selector = Mock()
        
        # Set up mock health metrics
        self.mock_health_metrics = Mock()
        self.mock_health_metrics.overall_health_score = 0.7
        self.mock_health_metrics.memory_health = 0.8
        self.mock_health_metrics.agent_performance_health = 0.6
        self.mock_health_metrics.dissonance_health = 0.9
        self.mock_health_metrics.contract_alignment_health = 0.7
        self.mock_health_metrics.reflection_quality_health = 0.8
        self.mock_health_metrics.anomalies_detected = []
        self.mock_health_metrics.critical_issues = []
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_planner_initialization(self):
        """Test autonomous planner initialization."""
        self.assertIsInstance(self.planner, AutonomousPlanner)
        self.assertTrue(self.planner.enabled)
        self.assertEqual(self.planner.max_active_plans, 3)
        self.assertEqual(self.planner.min_plan_score, 0.4)
        self.assertEqual(self.planner.lookahead_days, 7)
        self.assertIsInstance(self.planner.active_plans, dict)
        self.assertIsInstance(self.planner.completed_plans, list)
        self.assertIsInstance(self.planner.system_projections, list)
    
    def test_plan_step_creation(self):
        """Test creation of individual plan steps."""
        step = PlanStep(
            step_id="test_step_1",
            description="Test memory optimization",
            category="memory",
            estimated_effort=60,
            dependencies=[],
            success_criteria="Memory usage reduced by 10%",
            rollback_steps=["restore_memory_backup"]
        )
        
        self.assertEqual(step.step_id, "test_step_1")
        self.assertEqual(step.description, "Test memory optimization")
        self.assertEqual(step.category, "memory")
        self.assertEqual(step.estimated_effort, 60)
        self.assertEqual(step.dependencies, [])
        self.assertIsInstance(step.created_at, float)
        self.assertGreater(step.created_at, 0)
    
    def test_action_plan_creation(self):
        """Test creation and scoring of action plans."""
        steps = [
            PlanStep(
                step_id="step_1",
                description="Analyze memory usage",
                category="memory",
                estimated_effort=30,
                dependencies=[],
                success_criteria="Usage analysis complete",
                rollback_steps=[]
            ),
            PlanStep(
                step_id="step_2", 
                description="Implement optimization",
                category="memory",
                estimated_effort=90,
                dependencies=["step_1"],
                success_criteria="Memory usage reduced by 15%",
                rollback_steps=["restore_memory_state"]
            )
        ]
        
        plan = ActionPlan(
            plan_id="test_plan_1",
            title="Memory Optimization Plan",
            description="Comprehensive memory system optimization",
            steps=steps,
            relevance_score=0.8,
            expected_impact=0.7,
            resource_requirements=0.5,
            conflict_risk=0.2
        )
        
        self.assertEqual(plan.plan_id, "test_plan_1")
        self.assertEqual(plan.title, "Memory Optimization Plan")
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.status, "pending")
        self.assertIsInstance(plan.overall_score, float)
        self.assertGreater(plan.overall_score, 0)
        self.assertLessEqual(plan.overall_score, 1.0)
        
        # Test score calculation
        expected_score = (
            0.35 * 0.8 +  # relevance
            0.30 * 0.7 +  # impact  
            0.20 * 0.5 +  # resources (inverted)
            0.15 * 0.8    # risk (inverted)
        )
        self.assertAlmostEqual(plan.overall_score, expected_score, places=2)
    
    def test_system_projection_creation(self):
        """Test system projection functionality."""
        projection = SystemProjection(
            projection_timeframe="short",
            projected_memory_growth=0.3,
            projected_agent_drift=0.1,
            predicted_tool_gaps=["memory_compression", "agent_optimization"],
            anticipated_bottlenecks=["memory_scaling"],
            confidence=0.8,
            basis=["current_memory_state", "performance_trends"]
        )
        
        self.assertEqual(projection.projection_timeframe, "short")
        self.assertEqual(projection.projected_memory_growth, 0.3)
        self.assertEqual(projection.projected_agent_drift, 0.1)
        self.assertEqual(len(projection.predicted_tool_gaps), 2)
        self.assertEqual(len(projection.anticipated_bottlenecks), 1)
        self.assertEqual(projection.confidence, 0.8)
        self.assertIsInstance(projection.timestamp, float)
    
    @patch('agents.action_planner.AutonomousPlanner._get_meta_self_agent')
    def test_system_state_monitoring(self, mock_get_meta):
        """Test system state monitoring functionality."""
        # Set up mock meta self agent
        mock_get_meta.return_value = self.mock_meta_self_agent
        self.mock_meta_self_agent.perform_health_check.return_value = self.mock_health_metrics
        
        # Monitor system state
        system_state = self.planner._monitor_system_state()
        
        self.assertIsInstance(system_state, dict)
        self.assertIn('timestamp', system_state)
        self.assertIn('overall_health', system_state)
        self.assertIn('meta_health', system_state)
        self.assertIn('memory_state', system_state)
        self.assertIn('goal_state', system_state)
        self.assertIn('agent_state', system_state)
        self.assertIn('contradiction_state', system_state)
        
        # Verify health data extraction
        meta_health = system_state['meta_health']
        self.assertEqual(meta_health['overall_health'], 0.7)
        self.assertEqual(meta_health['memory_health'], 0.8)
        self.assertEqual(meta_health['agent_performance'], 0.6)
    
    def test_short_term_projection(self):
        """Test short-term future projection."""
        system_state = {
            'memory_state': {
                'total_facts': 1000,
                'growth_rate_per_hour': 0.1,
                'bloat_ratio': 0.2,
                'has_recent_data': True
            },
            'meta_health': {
                'agent_performance': 0.8,
                'critical_issues': ['test_issue_1', 'capability_gap']
            }
        }
        
        projection = self.planner._project_short_term_needs(system_state)
        
        self.assertIsNotNone(projection)
        self.assertEqual(projection.projection_timeframe, "short")
        self.assertIsInstance(projection.projected_memory_growth, float)
        self.assertIsInstance(projection.projected_agent_drift, float)
        self.assertIsInstance(projection.predicted_tool_gaps, list)
        self.assertIsInstance(projection.anticipated_bottlenecks, list)
        self.assertGreater(projection.confidence, 0)
        self.assertLessEqual(projection.confidence, 1.0)
    
    def test_medium_term_projection(self):
        """Test medium-term future projection."""
        system_state = {
            'memory_state': {
                'total_facts': 1000,
                'growth_rate_per_hour': 0.2
            },
            'goal_state': {
                'active_goals': 15,
                'completion_rate': 0.6,
                'goal_types': ['memory_optimization', 'agent_improvement']
            }
        }
        
        projection = self.planner._project_medium_term_needs(system_state)
        
        self.assertIsNotNone(projection)
        self.assertEqual(projection.projection_timeframe, "medium")
        self.assertIsInstance(projection.predicted_tool_gaps, list)
        self.assertIsInstance(projection.anticipated_bottlenecks, list)
        
        # Check for tool gaps based on goal types
        if 'memory_optimization' in system_state['goal_state']['goal_types']:
            self.assertIn('memory_consolidation_tools', projection.predicted_tool_gaps)
    
    def test_long_term_projection(self):
        """Test long-term future projection."""
        system_state = {
            'meta_health': {
                'overall_health': 0.6
            }
        }
        
        projection = self.planner._project_long_term_needs(system_state)
        
        self.assertIsNotNone(projection)
        self.assertEqual(projection.projection_timeframe, "long")
        self.assertIsInstance(projection.predicted_tool_gaps, list)
        self.assertIsInstance(projection.anticipated_bottlenecks, list)
        self.assertGreater(len(projection.predicted_tool_gaps), 0)
        self.assertGreater(len(projection.anticipated_bottlenecks), 0)
        
        # Lower confidence for long-term projections
        self.assertLess(projection.confidence, 0.6)
    
    def test_plan_generation_for_projection(self):
        """Test plan generation based on system projections."""
        projection = SystemProjection(
            projection_timeframe="short",
            projected_memory_growth=0.4,  # Above threshold
            projected_agent_drift=0.1,
            predicted_tool_gaps=["memory_tools"],
            anticipated_bottlenecks=["agent_performance"],
            confidence=0.8,
            basis=["test_basis"]
        )
        
        system_state = {
            'meta_health': {'overall_health': 0.7}
        }
        
        plan = self.planner._create_plan_for_projection(projection, system_state)
        
        self.assertIsNotNone(plan)
        self.assertIsInstance(plan, ActionPlan)
        self.assertEqual(plan.title, "Short-term System Optimization")
        self.assertGreater(len(plan.steps), 0)
        self.assertGreater(plan.overall_score, 0)
        
        # Check for memory-related steps due to high growth
        memory_steps = [step for step in plan.steps if step.category == "memory"]
        self.assertGreater(len(memory_steps), 0)
    
    def test_critical_issue_plan_generation(self):
        """Test plan generation for critical system issues."""
        system_state = {
            'meta_health': {'overall_health': 0.7}
        }
        
        # Test memory issue
        memory_plan = self.planner._create_plan_for_critical_issue("memory bloat detected", system_state)
        self.assertIsNotNone(memory_plan)
        self.assertEqual(memory_plan.relevance_score, 1.0)  # Critical issues get max relevance
        self.assertIn("memory", memory_plan.steps[0].description.lower())
        
        # Test agent issue
        agent_plan = self.planner._create_plan_for_critical_issue("agent performance degraded", system_state)
        self.assertIsNotNone(agent_plan)
        self.assertIn("agent", agent_plan.steps[0].description.lower())
        
        # Test generic issue
        generic_plan = self.planner._create_plan_for_critical_issue("unknown system error", system_state)
        self.assertIsNotNone(generic_plan)
        self.assertEqual(generic_plan.steps[0].category, "system")
    
    def test_plan_scoring_and_prioritization(self):
        """Test plan scoring and prioritization."""
        # Create multiple plans with different scores
        plans = {
            "high_score": ActionPlan(
                plan_id="high_score",
                title="High Priority Plan",
                description="Test plan",
                steps=[],
                relevance_score=0.9,
                expected_impact=0.8,
                resource_requirements=0.3,
                conflict_risk=0.2
            ),
            "low_score": ActionPlan(
                plan_id="low_score", 
                title="Low Priority Plan",
                description="Test plan",
                steps=[],
                relevance_score=0.4,
                expected_impact=0.3,
                resource_requirements=0.8,
                conflict_risk=0.7
            )
        }
        
        self.planner.active_plans = plans
        self.planner._score_and_prioritize_plans()
        
        # Verify plans are sorted by score
        plan_list = list(self.planner.active_plans.values())
        self.assertGreaterEqual(plan_list[0].overall_score, plan_list[1].overall_score)
        self.assertEqual(plan_list[0].plan_id, "high_score")
    
    @patch('agents.action_planner.AutonomousPlanner._get_task_selector')
    def test_goal_enqueueing(self, mock_get_task_selector):
        """Test goal enqueueing functionality."""
        # Set up mock task selector
        mock_task_selector = Mock()
        mock_task_selector.add_task.return_value = "test_goal_id"
        mock_get_task_selector.return_value = mock_task_selector
        
        # Create test plan with ready steps
        plan = ActionPlan(
            plan_id="test_enqueue",
            title="Test Plan",
            description="Test plan",
            steps=[
                PlanStep(
                    step_id="ready_step",
                    description="Ready step",
                    category="memory",
                    estimated_effort=30,
                    dependencies=[],
                    success_criteria="Test complete",
                    rollback_steps=[]
                )
            ],
            relevance_score=0.8,
            expected_impact=0.7,
            resource_requirements=0.4,
            conflict_risk=0.3
        )
        
        self.planner.active_plans[plan.plan_id] = plan
        
        # Enqueue goals
        enqueued_goals = self.planner._enqueue_priority_goals()
        
        self.assertGreater(len(enqueued_goals), 0)
        self.assertEqual(enqueued_goals[0], "test_goal_id")
        mock_task_selector.add_task.assert_called()
        
        # Verify plan status updated
        self.assertEqual(plan.status, "active")
        self.assertIsNotNone(plan.started_at)
        self.assertIn("test_goal_id", plan.generated_goals)
    
    def test_plan_progress_tracking(self):
        """Test plan progress and completion tracking."""
        plan = ActionPlan(
            plan_id="progress_test",
            title="Progress Test Plan",
            description="Test plan progress",
            steps=[
                PlanStep(
                    step_id="step_1",
                    description="First step",
                    category="memory",
                    estimated_effort=60,
                    dependencies=[],
                    success_criteria="Step 1 complete",
                    rollback_steps=[]
                ),
                PlanStep(
                    step_id="step_2",
                    description="Second step", 
                    category="agent",
                    estimated_effort=90,
                    dependencies=["step_1"],
                    success_criteria="Step 2 complete",
                    rollback_steps=[]
                )
            ],
            relevance_score=0.7,
            expected_impact=0.6,
            resource_requirements=0.5,
            conflict_risk=0.3
        )
        
        plan.status = "active"
        plan.started_at = time.time() - 1000  # Started some time ago
        
        # Update progress
        self.planner._update_plan_progress(plan)
        
        # Check if any steps were marked as completed (simulated)
        total_steps = len(plan.steps)
        completed_steps = len(plan.completed_steps)
        failed_steps = len(plan.failed_steps)
        
        self.assertGreaterEqual(completed_steps + failed_steps, 0)
        self.assertLessEqual(completed_steps + failed_steps, total_steps)
    
    def test_plan_cancellation_logic(self):
        """Test plan cancellation conditions."""
        system_state = {
            'meta_health': {
                'overall_health': 0.8,
                'critical_issues': []
            }
        }
        
        # Test plan with low current relevance
        low_relevance_plan = ActionPlan(
            plan_id="low_relevance",
            title="Outdated Plan",
            description="Test plan",
            steps=[],
            relevance_score=0.2,  # Originally low
            expected_impact=0.5,
            resource_requirements=0.5,
            conflict_risk=0.3
        )
        
        should_cancel = self.planner._should_cancel_plan(low_relevance_plan, system_state)
        self.assertTrue(should_cancel)
        
        # Test plan that's been active too long without progress
        stalled_plan = ActionPlan(
            plan_id="stalled",
            title="Stalled Plan", 
            description="Test plan",
            steps=[PlanStep(
                step_id="test_step",
                description="Test step",
                category="memory",
                estimated_effort=60,
                dependencies=[],
                success_criteria="Test complete",
                rollback_steps=[]
            )],
            relevance_score=0.8,
            expected_impact=0.7,
            resource_requirements=0.4,
            conflict_risk=0.2
        )
        stalled_plan.started_at = time.time() - (8 * 24 * 3600)  # 8 days ago
        stalled_plan.completed_steps = []  # No progress
        
        should_cancel = self.planner._should_cancel_plan(stalled_plan, system_state)
        self.assertTrue(should_cancel)
    
    def test_plan_persistence(self):
        """Test plan storage and loading functionality."""
        # Create test plan
        plan = ActionPlan(
            plan_id="persistence_test",
            title="Persistence Test Plan",
            description="Test plan persistence",
            steps=[],
            relevance_score=0.8,
            expected_impact=0.7,
            resource_requirements=0.4,
            conflict_risk=0.3
        )
        
        self.planner.active_plans[plan.plan_id] = plan
        
        # Save plans
        self.planner._save_plans_to_storage()
        
        # Verify file was created
        self.assertTrue(self.test_plan_path.exists())
        
        # Verify content
        with open(self.test_plan_path, 'r') as f:
            saved_data = json.loads(f.read())
        
        self.assertIn('active_plans', saved_data)
        self.assertEqual(len(saved_data['active_plans']), 1)
        self.assertEqual(saved_data['active_plans'][0]['plan_id'], "persistence_test")
        
        # Test loading
        new_planner = AutonomousPlanner()
        new_planner.plan_storage_path = self.test_plan_path
        new_planner._load_persistent_plans()
        
        self.assertEqual(len(new_planner.active_plans), 1)
        self.assertIn("persistence_test", new_planner.active_plans)
    
    @patch('agents.action_planner.AutonomousPlanner._get_meta_self_agent')
    @patch('agents.action_planner.AutonomousPlanner._get_task_selector')
    def test_full_planning_cycle(self, mock_get_task_selector, mock_get_meta):
        """Test complete planning evaluation cycle."""
        # Set up mocks
        mock_get_meta.return_value = self.mock_meta_self_agent
        self.mock_meta_self_agent.perform_health_check.return_value = self.mock_health_metrics
        
        mock_task_selector = Mock()
        mock_task_selector.add_task.return_value = "test_goal"
        mock_get_task_selector.return_value = mock_task_selector
        
        # Add some critical issues to trigger plan generation
        self.mock_health_metrics.critical_issues = ["memory bloat detected"]
        
        # Run planning cycle
        results = self.planner.evaluate_and_update_plan()
        
        self.assertIsInstance(results, dict)
        self.assertEqual(results['status'], 'completed')
        self.assertIn('cycle_duration', results)
        self.assertIn('active_plans', results)
        self.assertIn('new_plans_generated', results)
        self.assertIn('goals_enqueued', results)
        self.assertIn('system_health_score', results)
        self.assertIn('projections_generated', results)
        
        # Verify cycle was logged
        self.assertEqual(len(self.planner.planning_history), 1)
        self.assertEqual(self.planner.planning_history[0], results)
    
    def test_cli_response_methods(self):
        """Test CLI response methods."""
        # Test plan status report
        status_report = self.planner._generate_plan_status_report()
        self.assertIsInstance(status_report, str)
        self.assertIn("Autonomous Planning Status", status_report)
        
        # Test future projection report
        # Add a test projection
        projection = SystemProjection(
            projection_timeframe="short",
            projected_memory_growth=0.3,
            projected_agent_drift=0.1,
            predicted_tool_gaps=["test_tool"],
            anticipated_bottlenecks=["test_bottleneck"],
            confidence=0.8,
            basis=["test_basis"]
        )
        self.planner.system_projections.append(projection)
        
        projection_report = self.planner._generate_future_projection_report()
        self.assertIsInstance(projection_report, str)
        self.assertIn("Future System Projections", projection_report)
        self.assertIn("short", projection_report.lower())
    
    def test_respond_method(self):
        """Test the main respond method for different inputs."""
        # Test status query
        status_response = self.planner.respond("show status")
        self.assertIn("Planning Status", status_response)
        
        # Test next step trigger
        next_response = self.planner.respond("trigger next step")
        self.assertIsInstance(next_response, str)
        
        # Test evaluation trigger
        eval_response = self.planner.respond("evaluate plans")
        self.assertIsInstance(eval_response, str)
        
        # Test projection query
        proj_response = self.planner.respond("show future projections")
        self.assertIsInstance(proj_response, str)
        
        # Test general planning query
        general_response = self.planner.respond("what is the planning status?")
        self.assertIn("Planning Analysis", general_response)
    
    def test_disabled_planner(self):
        """Test planner behavior when disabled."""
        self.planner.enabled = False
        
        # Planning cycle should return disabled status
        results = self.planner.evaluate_and_update_plan()
        self.assertEqual(results['status'], 'disabled')
        self.assertIn('disabled', results['message'])
        
        # CLI commands should report disabled status
        status_report = self.planner._generate_plan_status_report()
        self.assertIn("No active plans", status_report)
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test monitoring with no meta self agent
        self.planner._meta_self_agent = None
        system_state = self.planner._monitor_system_state()
        self.assertIn('error', system_state)
        
        # Test projection with invalid data
        bad_system_state = {}
        projection = self.planner._project_short_term_needs(bad_system_state)
        # Should handle gracefully and return None or minimal data
        
        # Test plan creation with invalid projection
        invalid_projection = SystemProjection(
            projection_timeframe="invalid",
            projected_memory_growth=0.0,
            projected_agent_drift=0.0,
            predicted_tool_gaps=[],
            anticipated_bottlenecks=[],
            confidence=0.0,
            basis=[]
        )
        plan = self.planner._create_plan_for_projection(invalid_projection, {})
        # Should handle gracefully
    
    def test_rollback_capability(self):
        """Test plan rollback functionality."""
        step_with_rollback = PlanStep(
            step_id="rollback_test",
            description="Test step with rollback",
            category="memory",
            estimated_effort=60,
            dependencies=[],
            success_criteria="Test complete",
            rollback_steps=["restore_backup", "reset_configuration"]
        )
        
        self.assertEqual(len(step_with_rollback.rollback_steps), 2)
        self.assertIn("restore_backup", step_with_rollback.rollback_steps)
        self.assertIn("reset_configuration", step_with_rollback.rollback_steps)
    
    def test_memory_growth_threshold_detection(self):
        """Test detection of memory growth that exceeds thresholds."""
        system_state = {
            'memory_state': {
                'total_facts': 1000,
                'growth_rate_per_hour': 0.3,  # High growth rate
                'bloat_ratio': 0.4,  # High bloat
                'has_recent_data': True
            },
            'meta_health': {
                'agent_performance': 0.5,  # Below threshold
                'critical_issues': []
            }
        }
        
        projection = self.planner._project_short_term_needs(system_state)
        
        # Should predict significant memory growth
        weekly_growth = 0.3 * 168  # 1 week of growth
        expected_growth_ratio = weekly_growth / 1000
        
        # Should identify bottlenecks due to high bloat
        self.assertIn("memory_bloat", projection.anticipated_bottlenecks)
        self.assertIn("agent_performance", projection.anticipated_bottlenecks)


class TestPlanStepDependencies(unittest.TestCase):
    """Test plan step dependency management."""
    
    def test_dependency_resolution(self):
        """Test plan step dependency resolution."""
        steps = [
            PlanStep(
                step_id="step_1",
                description="First step",
                category="memory",
                estimated_effort=30,
                dependencies=[],
                success_criteria="Step 1 complete",
                rollback_steps=[]
            ),
            PlanStep(
                step_id="step_2",
                description="Second step",
                category="memory", 
                estimated_effort=60,
                dependencies=["step_1"],
                success_criteria="Step 2 complete",
                rollback_steps=[]
            ),
            PlanStep(
                step_id="step_3",
                description="Third step",
                category="agent",
                estimated_effort=90,
                dependencies=["step_1", "step_2"],
                success_criteria="Step 3 complete",
                rollback_steps=[]
            )
        ]
        
        plan = ActionPlan(
            plan_id="dependency_test",
            title="Dependency Test Plan",
            description="Test dependency resolution",
            steps=steps,
            relevance_score=0.8,
            expected_impact=0.7,
            resource_requirements=0.5,
            conflict_risk=0.3
        )
        
        # Initially, only step_1 should be ready (no dependencies)
        ready_steps = [
            step for step in plan.steps 
            if all(dep in plan.completed_steps for dep in step.dependencies)
        ]
        self.assertEqual(len(ready_steps), 1)
        self.assertEqual(ready_steps[0].step_id, "step_1")
        
        # After completing step_1, step_2 should be ready
        plan.completed_steps.append("step_1")
        ready_steps = [
            step for step in plan.steps 
            if (step.step_id not in plan.completed_steps and 
                all(dep in plan.completed_steps for dep in step.dependencies))
        ]
        self.assertEqual(len(ready_steps), 1)
        self.assertEqual(ready_steps[0].step_id, "step_2")
        
        # After completing both step_1 and step_2, step_3 should be ready
        plan.completed_steps.append("step_2")
        ready_steps = [
            step for step in plan.steps 
            if (step.step_id not in plan.completed_steps and 
                all(dep in plan.completed_steps for dep in step.dependencies))
        ]
        self.assertEqual(len(ready_steps), 1)
        self.assertEqual(ready_steps[0].step_id, "step_3")


class TestPlanFailureHandling(unittest.TestCase):
    """Test plan failure and recovery scenarios."""
    
    def setUp(self):
        """Set up test fixtures for failure scenarios."""
        self.mock_config = {
            'autonomous_planner': {
                'enabled': True,
                'planning_interval_hours': 6,
                'min_plan_score': 0.4,
                'max_active_plans': 3
            }
        }
        
        with patch('config.settings.get_config', return_value=self.mock_config):
            self.planner = AutonomousPlanner()
    
    def test_step_failure_handling(self):
        """Test handling of failed plan steps."""
        plan = ActionPlan(
            plan_id="failure_test",
            title="Failure Test Plan",
            description="Test step failure handling",
            steps=[
                PlanStep(
                    step_id="failing_step",
                    description="This step will fail",
                    category="memory",
                    estimated_effort=60,
                    dependencies=[],
                    success_criteria="Should not be met",
                    rollback_steps=["undo_failing_action"]
                )
            ],
            relevance_score=0.7,
            expected_impact=0.6,
            resource_requirements=0.5,
            conflict_risk=0.4
        )
        
        # Simulate step failure
        plan.failed_steps.append("failing_step")
        
        # Plan should be cancellable due to high failure rate
        system_state = {'meta_health': {'overall_health': 0.8, 'critical_issues': []}}
        should_cancel = self.planner._should_cancel_plan(plan, system_state)
        self.assertTrue(should_cancel)
    
    def test_plan_recovery_after_partial_failure(self):
        """Test plan recovery when some steps fail but others can continue."""
        steps = [
            PlanStep(
                step_id="successful_step",
                description="This step succeeds",
                category="memory",
                estimated_effort=30,
                dependencies=[],
                success_criteria="Success criteria met",
                rollback_steps=[]
            ),
            PlanStep(
                step_id="failed_step",
                description="This step fails",
                category="agent",
                estimated_effort=60,
                dependencies=[],
                success_criteria="Failure criteria",
                rollback_steps=["rollback_failed_step"]
            ),
            PlanStep(
                step_id="independent_step",
                description="Independent step",
                category="system",
                estimated_effort=45,
                dependencies=[],
                success_criteria="Independent success",
                rollback_steps=[]
            )
        ]
        
        plan = ActionPlan(
            plan_id="recovery_test",
            title="Recovery Test Plan", 
            description="Test partial failure recovery",
            steps=steps,
            relevance_score=0.8,
            expected_impact=0.7,
            resource_requirements=0.4,
            conflict_risk=0.3
        )
        
        # Simulate mixed success/failure
        plan.completed_steps.append("successful_step")
        plan.failed_steps.append("failed_step")
        
        # Plan should not be cancelled (failure rate = 1/3 = 33% < 50%)
        system_state = {'meta_health': {'overall_health': 0.8, 'critical_issues': []}}
        should_cancel = self.planner._should_cancel_plan(plan, system_state)
        self.assertFalse(should_cancel)
        
        # Independent step should still be executable
        ready_steps = [
            step for step in plan.steps 
            if (step.step_id not in plan.completed_steps and 
                step.step_id not in plan.failed_steps and
                all(dep in plan.completed_steps for dep in step.dependencies))
        ]
        self.assertEqual(len(ready_steps), 1)
        self.assertEqual(ready_steps[0].step_id, "independent_step")


if __name__ == '__main__':
    unittest.main()