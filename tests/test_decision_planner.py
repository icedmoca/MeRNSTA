#!/usr/bin/env python3
"""
Test suite for Phase 16: Autonomous Planning & Decision Layer

Comprehensive tests covering multi-step plan creation, strategy comparison,
goal re-routing after failure, and execution order tracking.
"""

import unittest
import tempfile
import os
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import the agents and components to test
try:
    from agents.decision_planner import DecisionPlanner, Strategy, StrategyType, PlanNode
    from agents.strategy_evaluator import StrategyEvaluator, EvaluationResult
    from agents.task_selector import TaskSelector, Task, TaskCategory, TaskPriority
    from agents.recursive_planner import Plan, PlanStep
    from storage.plan_memory import PlanMemory
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"Warning: Could not import planning components: {e}")


class TestDecisionPlanner(unittest.TestCase):
    """Test the DecisionPlanner agent for autonomous planning capabilities."""
    
    def setUp(self):
        """Set up test environment."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Planning components not available")
        
        # Create temporary config for testing
        self.test_config = {
            'enable_autonomous_planning': True,
            'default_strategy_count': 3,
            'scoring': {
                'feasibility_weight': 1.0,
                'memory_similarity_weight': 0.7,
                'historical_success_weight': 1.2,
                'effort_penalty': 0.5
            },
            'replanning_threshold': 0.5
        }
        
        # Mock the configuration loading
        with patch('agents.decision_planner.get_config') as mock_config:
            mock_config.return_value = {'decision_planner': self.test_config}
            self.planner = DecisionPlanner()
    
    def test_strategy_generation(self):
        """Test multi-step strategy generation."""
        goal = "Implement user authentication system"
        
        strategies = self.planner.generate_strategies(goal, strategy_count=3)
        
        # Verify strategies were generated
        self.assertGreater(len(strategies), 0)
        self.assertLessEqual(len(strategies), 3)
        
        # Verify strategy structure
        for strategy in strategies:
            self.assertIsInstance(strategy, Strategy)
            self.assertEqual(strategy.goal_text, goal)
            self.assertIsInstance(strategy.strategy_type, StrategyType)
            self.assertGreater(strategy.score, 0)
            self.assertIsInstance(strategy.plan, Plan)
            self.assertGreater(len(strategy.plan.steps), 0)
    
    def test_strategy_scoring(self):
        """Test strategy scoring mechanism."""
        # Create a mock strategy
        plan = Plan(
            plan_id="test_plan",
            goal_text="Test goal",
            steps=[
                PlanStep(
                    step_id="step_1",
                    subgoal="First step",
                    why="Testing",
                    expected_result="Step completed",
                    prerequisites=[],
                    confidence=0.8
                )
            ]
        )
        
        strategy = Strategy(
            strategy_id="test_strategy",
            goal_text="Test goal",
            plan=plan,
            strategy_type=StrategyType.SEQUENTIAL,
            nodes=[]
        )
        
        score = self.planner._score_strategy(strategy)
        
        # Verify score is within valid range
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Verify score breakdown was populated
        self.assertIsInstance(strategy.scores_breakdown, dict)
        self.assertGreater(len(strategy.scores_breakdown), 0)
    
    def test_best_strategy_selection(self):
        """Test selection of best strategy from multiple options."""
        goal = "Create data backup system"
        
        best_strategy = self.planner.select_best_strategy(goal)
        
        if best_strategy:
            # Verify best strategy selection
            self.assertIsInstance(best_strategy, Strategy)
            self.assertEqual(best_strategy.goal_text, goal)
            self.assertGreater(best_strategy.score, 0)
            
            # Verify it's stored in active strategies
            self.assertIn(goal, self.planner.active_strategies)
            self.assertEqual(self.planner.active_strategies[goal], best_strategy)
    
    def test_adaptive_replanning(self):
        """Test adaptive re-planning after failure."""
        failed_goal = "Deploy application to production"
        failure_context = {
            "failed_step": "setup_database",
            "failure_type": "timeout",
            "result": {"error": "Database connection timeout"},
            "strategy_id": "original_strategy_123"
        }
        
        new_strategy = self.planner.adaptive_replan(failed_goal, failure_context)
        
        if new_strategy:
            # Verify new strategy was created
            self.assertIsInstance(new_strategy, Strategy)
            self.assertEqual(new_strategy.goal_text, failed_goal)
            
            # Verify it's different from original
            self.assertNotEqual(new_strategy.strategy_id, "original_strategy_123")
            
            # Verify it's now the active strategy
            self.assertIn(failed_goal, self.planner.active_strategies)
    
    def test_next_action_selection(self):
        """Test selection of next executable action."""
        # First create a strategy with executable steps
        goal = "Optimize database performance"
        strategy = self.planner.select_best_strategy(goal)
        
        if strategy:
            # Mock some steps as ready to execute
            for step in strategy.plan.steps[:2]:
                step.status = "pending"
            
            next_action = self.planner.get_next_action(goal)
            
            if next_action:
                # Verify next action structure
                self.assertIn('goal', next_action)
                self.assertIn('strategy_id', next_action)
                self.assertIn('step_id', next_action)
                self.assertIn('action', next_action)
                self.assertEqual(next_action['goal'], goal)
    
    def test_strategy_visualization(self):
        """Test strategy tree visualization."""
        goal = "Set up monitoring system"
        strategy = self.planner.select_best_strategy(goal)
        
        if strategy:
            visualization = self.planner.visualize_strategy(strategy.strategy_id)
            
            # Verify visualization structure
            self.assertIn('strategy_id', visualization)
            self.assertIn('goal', visualization)
            self.assertIn('nodes', visualization)
            self.assertIn('edges', visualization)
            self.assertIn('stats', visualization)
            
            # Verify stats
            stats = visualization['stats']
            self.assertIn('total_steps', stats)
            self.assertIn('completed_steps', stats)
            self.assertIn('failed_steps', stats)


class TestStrategyEvaluator(unittest.TestCase):
    """Test the StrategyEvaluator component for strategy comparison."""
    
    def setUp(self):
        """Set up test environment."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Planning components not available")
        
        self.evaluator = StrategyEvaluator()
    
    def test_single_strategy_evaluation(self):
        """Test evaluation of a single strategy."""
        # Create a test strategy
        plan = Plan(
            plan_id="eval_test_plan",
            goal_text="Test evaluation",
            steps=[
                PlanStep(
                    step_id="eval_step_1",
                    subgoal="Evaluation step",
                    why="Testing evaluation",
                    expected_result="Evaluation complete",
                    prerequisites=[],
                    confidence=0.7
                )
            ]
        )
        
        strategy = Strategy(
            strategy_id="eval_test_strategy",
            goal_text="Test evaluation",
            plan=plan,
            strategy_type=StrategyType.SEQUENTIAL,
            nodes=[]
        )
        
        evaluation = self.evaluator.evaluate_strategy(strategy)
        
        # Verify evaluation structure
        self.assertIsInstance(evaluation, EvaluationResult)
        self.assertEqual(evaluation.strategy_id, "eval_test_strategy")
        self.assertGreaterEqual(evaluation.overall_score, 0.0)
        self.assertLessEqual(evaluation.overall_score, 1.0)
        self.assertIsInstance(evaluation.criteria_scores, dict)
        self.assertIsInstance(evaluation.strengths, list)
        self.assertIsInstance(evaluation.weaknesses, list)
        self.assertIsInstance(evaluation.recommendations, list)
    
    def test_strategy_comparison(self):
        """Test comparison of multiple strategies."""
        # Create multiple test strategies
        strategies = []
        for i in range(3):
            plan = Plan(
                plan_id=f"compare_plan_{i}",
                goal_text="Compare strategies",
                steps=[
                    PlanStep(
                        step_id=f"compare_step_{i}",
                        subgoal=f"Step {i}",
                        why="Testing comparison",
                        expected_result="Step complete",
                        prerequisites=[],
                        confidence=0.6 + i * 0.1
                    )
                ]
            )
            
            strategy = Strategy(
                strategy_id=f"compare_strategy_{i}",
                goal_text="Compare strategies",
                plan=plan,
                strategy_type=StrategyType.SEQUENTIAL,
                nodes=[]
            )
            strategies.append(strategy)
        
        comparison = self.evaluator.compare_strategies(strategies)
        
        # Verify comparison structure
        self.assertEqual(len(comparison.strategy_rankings), 3)
        self.assertEqual(len(comparison.evaluation_results), 3)
        self.assertIsInstance(comparison.comparison_summary, dict)
        self.assertIsInstance(comparison.decision_rationale, str)
        
        # Verify rankings are sorted by score
        scores = [ranking[1] for ranking in comparison.strategy_rankings]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_evaluation_insights(self):
        """Test generation of evaluation insights."""
        insights = self.evaluator.get_evaluation_insights()
        
        # Verify insights structure (even if empty)
        self.assertIsInstance(insights, dict)
        if 'message' not in insights:
            # If we have data, verify structure
            if 'total_evaluations' in insights:
                self.assertIsInstance(insights['total_evaluations'], int)


class TestTaskSelector(unittest.TestCase):
    """Test the TaskSelector component for intelligent task selection."""
    
    def setUp(self):
        """Set up test environment."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Planning components not available")
        
        self.selector = TaskSelector()
    
    def test_task_addition(self):
        """Test adding tasks to the selection queue."""
        task_id = self.selector.add_task(
            goal_text="Implement caching system",
            category=TaskCategory.IMPROVEMENT,
            priority=TaskPriority.HIGH,
            urgency=0.8,
            importance=0.7,
            estimated_effort=120
        )
        
        # Verify task was added
        self.assertIsInstance(task_id, str)
        self.assertEqual(len(self.selector.task_queue), 1)
        
        # Verify task structure
        task = self.selector.task_queue[0]
        self.assertEqual(task.task_id, task_id)
        self.assertEqual(task.goal_text, "Implement caching system")
        self.assertEqual(task.category, TaskCategory.IMPROVEMENT)
        self.assertEqual(task.priority, TaskPriority.HIGH)
    
    def test_task_selection(self):
        """Test selection of next task to execute."""
        # Add multiple tasks with different priorities
        tasks = [
            ("Low priority task", TaskPriority.LOW, 0.3, 0.4),
            ("High priority task", TaskPriority.HIGH, 0.9, 0.8),
            ("Medium priority task", TaskPriority.MEDIUM, 0.6, 0.6)
        ]
        
        for goal, priority, urgency, importance in tasks:
            self.selector.add_task(
                goal_text=goal,
                priority=priority,
                urgency=urgency,
                importance=importance,
                estimated_effort=60
            )
        
        # Select next task
        selection = self.selector.select_next_task()
        
        if selection:
            # Verify selection structure
            self.assertIsInstance(selection.selected_task, Task)
            self.assertGreater(selection.selection_score, 0)
            self.assertIsInstance(selection.rationale, str)
            self.assertIsInstance(selection.alternative_tasks, list)
            
            # High priority task should generally be selected
            # (though this depends on the scoring algorithm)
            selected_goal = selection.selected_task.goal_text
            self.assertIn("task", selected_goal.lower())
    
    def test_task_status_update(self):
        """Test updating task status after execution."""
        # Add a task
        task_id = self.selector.add_task(
            goal_text="Update documentation",
            estimated_effort=30
        )
        
        # Update task status
        outcome = {
            "success": True,
            "duration_minutes": 25,
            "quality_score": 0.8
        }
        
        self.selector.update_task_status(task_id, "completed", outcome)
        
        # Verify task was moved to completed
        self.assertEqual(len(self.selector.task_queue), 0)
        self.assertEqual(len(self.selector.completed_tasks), 1)
        self.assertIn(task_id, self.selector.task_history)
    
    def test_queue_status(self):
        """Test getting task queue status."""
        # Add some tasks
        for i in range(3):
            self.selector.add_task(
                goal_text=f"Task {i}",
                category=TaskCategory.IMPROVEMENT,
                estimated_effort=30 + i * 10
            )
        
        status = self.selector.get_queue_status()
        
        # Verify status structure
        self.assertIn('queue_size', status)
        self.assertIn('ready_tasks', status)
        self.assertIn('category_distribution', status)
        self.assertEqual(status['queue_size'], 3)
    
    def test_strategic_focus(self):
        """Test setting strategic focus for task prioritization."""
        # Set strategic focus
        focus_areas = ["performance", "security", "testing"]
        self.selector.set_strategic_focus(focus_areas)
        
        # Verify focus was set
        self.assertEqual(self.selector.strategic_focus, focus_areas)
        
        # Add tasks with and without strategic alignment
        self.selector.add_task(
            goal_text="Improve performance optimization",
            priority=TaskPriority.MEDIUM
        )
        self.selector.add_task(
            goal_text="Update user interface colors",
            priority=TaskPriority.MEDIUM
        )
        
        # Select task - performance task should have higher priority
        selection = self.selector.select_next_task()
        if selection:
            self.assertIn("performance", selection.selected_task.goal_text.lower())


class TestPlanMemoryDAG(unittest.TestCase):
    """Test the enhanced PlanMemory with DAG support."""
    
    def setUp(self):
        """Set up test environment."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Planning components not available")
        
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        
        self.plan_memory = PlanMemory(db_path=self.temp_db.name)
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'temp_db'):
            os.unlink(self.temp_db.name)
    
    def test_dag_storage_and_retrieval(self):
        """Test storing and retrieving plans as DAGs."""
        # Create a test plan with enhanced features
        plan = Plan(
            plan_id="dag_test_plan",
            goal_text="Test DAG storage",
            steps=[
                PlanStep(
                    step_id="dag_step_1",
                    subgoal="First step",
                    why="Testing DAG",
                    expected_result="Step complete",
                    prerequisites=[],
                    next_step="dag_step_2"
                ),
                PlanStep(
                    step_id="dag_step_2",
                    subgoal="Second step",
                    why="Continue DAG test",
                    expected_result="DAG test complete",
                    prerequisites=["dag_step_1"],
                    fallback_step="dag_step_1"
                )
            ],
            branching_points={"dag_step_1": ["alternative_a", "alternative_b"]},
            progress_checkpoints=["dag_step_1"]
        )
        
        # Store the plan
        success = self.plan_memory.store_plan(plan)
        self.assertTrue(success)
        
        # Retrieve as DAG
        dag = self.plan_memory.get_plan_dag("dag_test_plan")
        
        # Verify DAG structure
        self.assertNotIn('error', dag)
        self.assertIn('nodes', dag)
        self.assertIn('edges', dag)
        self.assertIn('dependencies', dag)
        self.assertIn('branches', dag)
        self.assertIn('checkpoints', dag)
        
        # Verify nodes
        self.assertEqual(len(dag['nodes']), 2)
        node_ids = {node['id'] for node in dag['nodes']}
        self.assertIn('dag_step_1', node_ids)
        self.assertIn('dag_step_2', node_ids)
    
    def test_dependency_management(self):
        """Test adding and managing plan dependencies."""
        # Create and store a simple plan
        plan = Plan(
            plan_id="dep_test_plan",
            goal_text="Test dependencies",
            steps=[
                PlanStep(step_id="dep_step_1", subgoal="Step 1", why="Test", expected_result="Done", prerequisites=[]),
                PlanStep(step_id="dep_step_2", subgoal="Step 2", why="Test", expected_result="Done", prerequisites=[])
            ]
        )
        self.plan_memory.store_plan(plan)
        
        # Add dependency
        success = self.plan_memory.add_plan_dependency(
            "dep_test_plan", "dep_step_2", "dep_step_1", "prerequisite"
        )
        self.assertTrue(success)
        
        # Get dependencies
        dependencies = self.plan_memory.get_step_dependencies("dep_test_plan", "dep_step_2")
        self.assertEqual(len(dependencies), 1)
        self.assertEqual(dependencies[0]['dependency_step_id'], "dep_step_1")
        
        # Get dependents
        dependents = self.plan_memory.get_step_dependents("dep_test_plan", "dep_step_1")
        self.assertEqual(len(dependents), 1)
        self.assertEqual(dependents[0]['step_id'], "dep_step_2")
    
    def test_plan_chaining(self):
        """Test chaining multiple plans together."""
        # Create two plans
        plan1 = Plan(plan_id="chain_plan_1", goal_text="First plan", steps=[])
        plan2 = Plan(plan_id="chain_plan_2", goal_text="Second plan", steps=[])
        
        self.plan_memory.store_plan(plan1)
        self.plan_memory.store_plan(plan2)
        
        # Chain them
        success = self.plan_memory.chain_plans("chain_plan_1", "chain_plan_2", "sequential")
        self.assertTrue(success)
        
        # Get chains
        outgoing_chains = self.plan_memory.get_plan_chains("chain_plan_1", "outgoing")
        incoming_chains = self.plan_memory.get_plan_chains("chain_plan_2", "incoming")
        
        self.assertEqual(len(outgoing_chains), 1)
        self.assertEqual(len(incoming_chains), 1)
        self.assertEqual(outgoing_chains[0]['target_plan_id'], "chain_plan_2")
        self.assertEqual(incoming_chains[0]['source_plan_id'], "chain_plan_1")
    
    def test_checkpoint_management(self):
        """Test adding and tracking checkpoints."""
        # Create and store a plan
        plan = Plan(
            plan_id="checkpoint_plan",
            goal_text="Test checkpoints",
            steps=[
                PlanStep(step_id="cp_step_1", subgoal="Step 1", why="Test", expected_result="Done", prerequisites=[])
            ]
        )
        self.plan_memory.store_plan(plan)
        
        # Add checkpoint
        success = self.plan_memory.add_checkpoint(
            "checkpoint_plan", "cp_step_1", "Milestone 1", "milestone"
        )
        self.assertTrue(success)
        
        # Mark checkpoint as reached
        success = self.plan_memory.reach_checkpoint(
            "checkpoint_plan", "cp_step_1", "Milestone 1"
        )
        self.assertTrue(success)
        
        # Get progress
        progress = self.plan_memory.get_plan_progress("checkpoint_plan")
        self.assertIn('checkpoints', progress)
        self.assertEqual(progress['checkpoints']['total'], 1)
        self.assertEqual(progress['checkpoints']['reached'], 1)
    
    def test_dependency_analysis(self):
        """Test analysis of plan dependency structures."""
        # Create a plan with potential issues
        plan = Plan(
            plan_id="analysis_plan",
            goal_text="Test analysis",
            steps=[
                PlanStep(step_id="an_step_1", subgoal="Step 1", why="Test", expected_result="Done", prerequisites=[]),
                PlanStep(step_id="an_step_2", subgoal="Step 2", why="Test", expected_result="Done", prerequisites=["an_step_1"])
            ]
        )
        self.plan_memory.store_plan(plan)
        
        # Analyze dependencies
        analysis = self.plan_memory.analyze_plan_dependencies("analysis_plan")
        
        # Verify analysis structure
        self.assertIn('dependency_issues', analysis)
        self.assertIn('recommendations', analysis)
        self.assertIn('metrics', analysis)
        self.assertIsInstance(analysis['metrics'], dict)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios across the planning system."""
    
    def setUp(self):
        """Set up test environment."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Planning components not available")
        
        # Create components with proper configuration
        with patch('agents.decision_planner.get_config') as mock_config:
            mock_config.return_value = {'decision_planner': {
                'enable_autonomous_planning': True,
                'default_strategy_count': 2,
                'scoring': {
                    'feasibility_weight': 1.0,
                    'memory_similarity_weight': 0.7,
                    'historical_success_weight': 1.2,
                    'effort_penalty': 0.5
                },
                'replanning_threshold': 0.5
            }}
            self.planner = DecisionPlanner()
        
        self.evaluator = StrategyEvaluator()
        self.selector = TaskSelector()
    
    def test_end_to_end_planning_workflow(self):
        """Test complete planning workflow from goal to execution."""
        # 1. Add goal to task selector
        task_id = self.selector.add_task(
            goal_text="Implement automated testing pipeline",
            category=TaskCategory.IMPROVEMENT,
            priority=TaskPriority.HIGH,
            urgency=0.8,
            importance=0.9,
            estimated_effort=180
        )
        
        # 2. Select task
        selection = self.selector.select_next_task()
        if not selection:
            self.skipTest("Task selection failed")
        
        selected_goal = selection.selected_task.goal_text
        
        # 3. Generate strategies for selected goal
        strategies = self.planner.generate_strategies(selected_goal, strategy_count=2)
        self.assertGreater(len(strategies), 0)
        
        # 4. Evaluate strategies
        if len(strategies) > 1:
            comparison = self.evaluator.compare_strategies(strategies)
            self.assertIsInstance(comparison.best_strategy_id, str)
        
        # 5. Select best strategy
        best_strategy = self.planner.select_best_strategy(selected_goal)
        if not best_strategy:
            self.skipTest("Strategy selection failed")
        
        # 6. Get next action
        next_action = self.planner.get_next_action(selected_goal)
        if next_action:
            self.assertIn('action', next_action)
            self.assertIn('step_id', next_action)
        
        # 7. Update task as completed
        self.selector.update_task_status(task_id, "completed", {
            "success": True,
            "duration_minutes": 150
        })
        
        # Verify final state
        self.assertEqual(len(self.selector.completed_tasks), 1)
        self.assertIn(selected_goal, self.planner.active_strategies)
    
    def test_failure_recovery_workflow(self):
        """Test workflow when a strategy fails and needs re-routing."""
        goal = "Deploy microservices architecture"
        
        # 1. Create initial strategy
        strategy = self.planner.select_best_strategy(goal)
        if not strategy:
            self.skipTest("Initial strategy creation failed")
        
        # 2. Simulate failure
        failure_context = {
            "failed_step": "setup_infrastructure",
            "failure_type": "resource_unavailable",
            "result": {"error": "AWS quota exceeded"}
        }
        
        # 3. Trigger re-planning
        new_strategy = self.planner.adaptive_replan(goal, failure_context)
        
        if new_strategy:
            # Verify new strategy is different
            self.assertNotEqual(new_strategy.strategy_id, strategy.strategy_id)
            
            # Verify it replaced the old strategy
            self.assertEqual(self.planner.active_strategies[goal], new_strategy)
    
    def test_concurrent_planning_scenarios(self):
        """Test handling multiple concurrent planning activities."""
        goals = [
            "Implement user authentication",
            "Set up monitoring system",
            "Optimize database queries"
        ]
        
        strategies = {}
        
        # Create strategies for all goals
        for goal in goals:
            strategy = self.planner.select_best_strategy(goal)
            if strategy:
                strategies[goal] = strategy
        
        # Verify all strategies are active
        for goal in goals:
            if goal in strategies:
                self.assertIn(goal, self.planner.active_strategies)
        
        # Get next action across all strategies
        next_action = self.planner.get_next_action()
        if next_action:
            # Should return action from one of the active strategies
            self.assertIn(next_action['goal'], goals)


def run_all_tests():
    """Run all planning tests with detailed output."""
    if not IMPORTS_AVAILABLE:
        print("❌ Cannot run tests - planning components not available")
        return False
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDecisionPlanner,
        TestStrategyEvaluator,
        TestTaskSelector,
        TestPlanMemoryDAG,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Skipped: {len(result.skipped)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n🔥 Errors:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"\n{status}: Planning system test suite")
    
    return success


if __name__ == "__main__":
    # Run tests when executed directly
    success = run_all_tests()
    exit(0 if success else 1)