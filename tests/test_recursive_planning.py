#!/usr/bin/env python3
"""
Comprehensive tests for Phase 11: Recursive Planning System

Tests all components of the recursive planning capabilities including:
- RecursivePlanner (plan generation, scoring, execution)
- PlanMemory (storage, retrieval, similarity matching)
- IntentionModel (intention tracking, causal chains)
- SelfPromptGenerator (autonomous goal generation)
- CLI integration
"""

import pytest
import tempfile
import os
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import the recursive planning components
from agents.recursive_planner import RecursivePlanner, Plan, PlanStep
from storage.plan_memory import PlanMemory
from storage.intention_model import IntentionModel, IntentionRecord
from agents.self_prompter import SelfPromptGenerator

class TestPlanDataStructures:
    """Test the basic data structures for plans and intentions"""
    
    def test_plan_step_creation(self):
        """Test PlanStep creation and defaults"""
        step = PlanStep(
            step_id="test-step-1",
            subgoal="Test subgoal",
            why="Test reasoning",
            expected_result="Test result",
            prerequisites=["prereq1", "prereq2"]
        )
        
        assert step.step_id == "test-step-1"
        assert step.subgoal == "Test subgoal"
        assert step.why == "Test reasoning"
        assert step.expected_result == "Test result"
        assert step.prerequisites == ["prereq1", "prereq2"]
        assert step.status == "pending"
        assert step.confidence == 0.8
        assert step.resources_needed == []
    
    def test_plan_creation(self):
        """Test Plan creation and defaults"""
        steps = [
            PlanStep("step1", "Goal 1", "Because", "Result 1", []),
            PlanStep("step2", "Goal 2", "Because", "Result 2", ["Goal 1"])
        ]
        
        plan = Plan(
            plan_id="test-plan-1",
            goal_text="Test goal",
            steps=steps
        )
        
        assert plan.plan_id == "test-plan-1"
        assert plan.goal_text == "Test goal"
        assert len(plan.steps) == 2
        assert plan.plan_type == "sequential"
        assert plan.status == "draft"
        assert plan.confidence == 0.8
        assert plan.intention_chain == []
        assert plan.success_criteria == []
        assert plan.risk_factors == []

class TestRecursivePlanner:
    """Test the RecursivePlanner agent"""
    
    @pytest.fixture
    def planner(self):
        """Create a test planner instance"""
        with patch('agents.recursive_planner.RecursivePlanner._RecursivePlanner__plan_memory', None):
            with patch('agents.recursive_planner.RecursivePlanner._RecursivePlanner__intention_model', None):
                planner = RecursivePlanner()
                # Mock the memory systems to avoid database dependencies
                planner._plan_memory = Mock()
                planner._intention_model = Mock()
                return planner
    
    def test_planner_initialization(self, planner):
        """Test planner initialization"""
        assert planner.name == "recursive_planner"
        assert planner.max_recursion_depth >= 1
        assert planner.min_confidence_threshold >= 0.0
        assert planner.plan_similarity_threshold >= 0.0
    
    def test_plan_goal_basic(self, planner):
        """Test basic plan generation"""
        # Mock LLM response
        with patch.object(planner, '_generate_plan_with_llm') as mock_llm:
            mock_llm.return_value = json.dumps({
                "goal_text": "Test goal",
                "plan_type": "sequential",
                "steps": [
                    {
                        "subgoal": "Step 1",
                        "why": "First step",
                        "expected_result": "Result 1",
                        "prerequisites": [],
                        "resources_needed": ["resource1"]
                    }
                ],
                "success_criteria": ["Goal achieved"],
                "risk_factors": ["Risk 1"]
            })
            
            plan = planner.plan_goal("Test goal")
            
            assert plan.goal_text == "Test goal"
            assert len(plan.steps) == 1
            assert plan.steps[0].subgoal == "Step 1"
            assert plan.success_criteria == ["Goal achieved"]
            assert plan.risk_factors == ["Risk 1"]
    
    def test_plan_goal_with_recursion_limit(self, planner):
        """Test plan generation respects recursion limit"""
        planner.max_recursion_depth = 2
        
        plan = planner.plan_goal("Complex goal", depth=3)
        
        # Should create a simple plan due to depth limit
        assert plan is not None
        assert len(plan.steps) >= 1
    
    def test_score_plan_empty(self, planner):
        """Test scoring an empty plan"""
        empty_plan = Plan("test", "goal", [])
        score = planner.score_plan(empty_plan)
        assert score == 0.0
    
    def test_score_plan_basic(self, planner):
        """Test basic plan scoring"""
        steps = [
            PlanStep("1", "Clear subgoal", "Good reasoning here", "Clear expected result", []),
            PlanStep("2", "Another goal", "More reasoning", "Another result", ["Clear subgoal"])
        ]
        plan = Plan("test", "Test goal", steps)
        
        score = planner.score_plan(plan)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be decent score for well-formed plan
    
    def test_execute_plan_empty(self, planner):
        """Test executing an empty plan"""
        empty_plan = Plan("test", "goal", [])
        results = planner.execute_plan(empty_plan)
        
        assert results["success"] == False
        assert "error" in results
    
    def test_execute_plan_basic(self, planner):
        """Test basic plan execution"""
        steps = [
            PlanStep("1", "Simple task", "Easy to do", "Task completed", [])
        ]
        plan = Plan("test", "Test goal", steps)
        
        results = planner.execute_plan(plan)
        
        assert "plan_id" in results
        assert "execution_start" in results
        assert "completion_percentage" in results
        assert isinstance(results["steps_executed"], list)
        assert isinstance(results["steps_failed"], list)
    
    def test_reuse_similar_plan_none_available(self, planner):
        """Test plan reuse when no similar plans exist"""
        planner._plan_memory.get_similar_plans.return_value = []
        
        similar_plan = planner.reuse_similar_plan("Unique goal")
        assert similar_plan is None
    
    def test_reuse_similar_plan_available(self, planner):
        """Test plan reuse when similar plans exist"""
        mock_similar_plans = [{
            'plan_id': 'similar-1',
            'similarity_score': 0.8,
            'success_rate': 0.9
        }]
        
        mock_plan = Plan("similar-1", "Similar goal", [
            PlanStep("s1", "Similar step", "Because", "Result", [])
        ])
        
        planner._plan_memory.get_similar_plans.return_value = mock_similar_plans
        planner._plan_memory.get_plan_by_id.return_value = mock_plan
        
        similar_plan = planner.reuse_similar_plan("Very similar goal")
        assert similar_plan is not None

class TestPlanMemory:
    """Test the PlanMemory storage system"""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def plan_memory(self, temp_db):
        """Create a test plan memory instance"""
        return PlanMemory(db_path=temp_db)
    
    def test_initialization(self, plan_memory):
        """Test plan memory initialization"""
        assert os.path.exists(plan_memory.db_path)
        stats = plan_memory.get_plan_statistics()
        assert stats['total_plans'] == 0
    
    def test_store_and_retrieve_plan(self, plan_memory):
        """Test storing and retrieving a plan"""
        steps = [
            PlanStep("step1", "Test step", "Test why", "Test result", [])
        ]
        plan = Plan("test-plan-1", "Test goal", steps)
        
        # Store plan
        success = plan_memory.store_plan(plan)
        assert success == True
        
        # Retrieve plan
        retrieved = plan_memory.get_plan_by_id("test-plan-1")
        assert retrieved is not None
        
        if hasattr(retrieved, 'goal_text'):
            assert retrieved.goal_text == "Test goal"
        else:
            assert retrieved['goal_text'] == "Test goal"
    
    def test_get_similar_plans_exact_match(self, plan_memory):
        """Test finding exact matches for similar plans"""
        plan = Plan("exact-1", "Learn Python programming", [
            PlanStep("s1", "Read tutorial", "Learning foundation", "Knowledge gained", [])
        ])
        
        plan_memory.store_plan(plan)
        
        # Update plan status to make it findable
        import sqlite3
        conn = sqlite3.connect(plan_memory.db_path)
        conn.execute("UPDATE plans SET status = 'completed', execution_count = 1, success_count = 1 WHERE plan_id = ?", (plan.plan_id,))
        conn.commit()
        conn.close()
        
        similar = plan_memory.get_similar_plans("Learn Python programming")
        assert len(similar) > 0
        assert similar[0]['similarity_score'] == 1.0  # Exact match
    
    def test_record_plan_outcome(self, plan_memory):
        """Test recording plan execution outcomes"""
        plan = Plan("outcome-test", "Test goal", [])
        plan_memory.store_plan(plan)
        
        outcome = {
            "overall_success": True,
            "completion_percentage": 100.0,
            "execution_start": datetime.now().isoformat(),
            "execution_end": datetime.now().isoformat(),
            "steps_executed": [{"step_id": "s1", "subgoal": "Test"}],
            "steps_failed": [],
            "execution_log": [{"timestamp": datetime.now().isoformat(), "message": "Test"}]
        }
        
        success = plan_memory.record_plan_outcome("outcome-test", outcome)
        assert success == True
        
        # Check statistics updated
        stats = plan_memory.get_plan_statistics()
        assert stats['total_plans'] >= 1
        assert stats['total_executions'] >= 1
    
    def test_get_plans_by_status(self, plan_memory):
        """Test filtering plans by status"""
        plan1 = Plan("status-1", "Goal 1", [])
        plan2 = Plan("status-2", "Goal 2", [])
        
        plan_memory.store_plan(plan1)
        plan_memory.store_plan(plan2)
        
        # Update one plan to completed status
        import sqlite3
        conn = sqlite3.connect(plan_memory.db_path)
        conn.execute("UPDATE plans SET status = 'completed' WHERE plan_id = ?", (plan1.plan_id,))
        conn.commit()
        conn.close()
        
        completed_plans = plan_memory.get_plans_by_status('completed')
        assert len(completed_plans) >= 1
        
        draft_plans = plan_memory.get_plans_by_status('draft')
        assert len(draft_plans) >= 1
    
    def test_delete_plan(self, plan_memory):
        """Test plan deletion"""
        plan = Plan("delete-test", "To be deleted", [])
        plan_memory.store_plan(plan)
        
        # Verify it exists
        retrieved = plan_memory.get_plan_by_id("delete-test")
        assert retrieved is not None
        
        # Delete it
        success = plan_memory.delete_plan("delete-test")
        assert success == True
        
        # Verify it's gone
        retrieved = plan_memory.get_plan_by_id("delete-test")
        assert retrieved is None

class TestIntentionModel:
    """Test the IntentionModel system"""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def intention_model(self, temp_db):
        """Create a test intention model instance"""
        return IntentionModel(db_path=temp_db)
    
    def test_initialization(self, intention_model):
        """Test intention model initialization"""
        assert os.path.exists(intention_model.db_path)
        stats = intention_model.get_intention_statistics()
        assert stats['total_intentions'] == 0
    
    def test_record_intention(self, intention_model):
        """Test recording an intention"""
        intention_id = intention_model.record_intention(
            goal_id="goal-1",
            triggered_by="user_request",
            drive="problem_solving",
            importance=0.8,
            reflection_note="Solving an important problem"
        )
        
        assert intention_id != ""
        
        # Check statistics
        stats = intention_model.get_intention_statistics()
        assert stats['total_intentions'] == 1
        assert stats['active_count'] == 1
    
    def test_link_goals(self, intention_model):
        """Test linking goals in a causal chain"""
        # Record parent intention
        parent_id = intention_model.record_intention(
            goal_id="parent-goal",
            triggered_by="analysis",
            drive="understanding",
            importance=0.7,
            reflection_note="Need to understand the problem"
        )
        
        # Record child intention
        child_id = intention_model.record_intention(
            goal_id="child-goal",
            triggered_by="parent_goal",
            drive="problem_solving",
            importance=0.8,
            reflection_note="Solve the specific issue"
        )
        
        # Link them
        success = intention_model.link_goals("child-goal", "parent-goal", "Child derives from parent")
        assert success == True
    
    def test_trace_why_empty(self, intention_model):
        """Test tracing why for non-existent goal"""
        trace = intention_model.trace_why("non-existent")
        assert trace == []
    
    def test_trace_why_single_level(self, intention_model):
        """Test tracing why for single-level goal"""
        intention_model.record_intention(
            goal_id="single-goal",
            triggered_by="user_request",
            drive="task_completion",
            importance=0.6,
            reflection_note="User asked me to do this"
        )
        
        trace = intention_model.trace_why("single-goal")
        assert len(trace) == 1
        assert trace[0]['goal_id'] == "single-goal"
        assert trace[0]['drive'] == "task_completion"
    
    def test_trace_why_multi_level(self, intention_model):
        """Test tracing why for multi-level goal chain"""
        # Create a chain: grandparent -> parent -> child
        intention_model.record_intention(
            goal_id="grandparent",
            triggered_by="system_analysis",
            drive="self_improvement",
            importance=0.9,
            reflection_note="Need to improve overall performance"
        )
        
        intention_model.record_intention(
            goal_id="parent",
            triggered_by="grandparent_goal",
            drive="efficiency",
            importance=0.8,
            reflection_note="Make planning more efficient",
            parent_goal_id="grandparent"
        )
        
        intention_model.record_intention(
            goal_id="child",
            triggered_by="parent_goal",
            drive="optimization",
            importance=0.7,
            reflection_note="Optimize specific algorithm",
            parent_goal_id="parent"
        )
        
        trace = intention_model.trace_why("child")
        assert len(trace) == 3
        assert trace[0]['goal_id'] == "child"
        assert trace[1]['goal_id'] == "parent"
        assert trace[2]['goal_id'] == "grandparent"
    
    def test_trace_why_formatted(self, intention_model):
        """Test formatted why explanation"""
        intention_model.record_intention(
            goal_id="format-test",
            triggered_by="test",
            drive="learning",
            importance=0.8,
            reflection_note="Testing the formatting"
        )
        
        explanation = intention_model.trace_why_formatted("format-test")
        assert "Why am I doing this?" in explanation
        assert "Testing the formatting" in explanation
        assert "🎯" in explanation  # Should have emoji
    
    def test_mark_goal_fulfilled(self, intention_model):
        """Test marking a goal as fulfilled"""
        intention_model.record_intention(
            goal_id="fulfill-test",
            triggered_by="test",
            drive="task_completion",
            importance=0.5,
            reflection_note="Test goal"
        )
        
        success = intention_model.mark_goal_fulfilled("fulfill-test")
        assert success == True
        
        stats = intention_model.get_intention_statistics()
        assert stats['fulfilled_count'] == 1
        assert stats['active_count'] == 0

class TestSelfPromptGenerator:
    """Test the SelfPromptGenerator agent"""
    
    @pytest.fixture
    def self_prompter(self):
        """Create a test self-prompter instance"""
        prompter = SelfPromptGenerator()
        # Mock the external systems to avoid dependencies
        prompter._memory_system = Mock()
        prompter._plan_memory = Mock()
        prompter._intention_model = Mock()
        prompter._reflection_system = Mock()
        return prompter
    
    def test_initialization(self, self_prompter):
        """Test self-prompter initialization"""
        assert self_prompter.name == "self_prompter"
        assert self_prompter.analysis_lookback_days > 0
        assert self_prompter.max_goals_per_session > 0
        assert hasattr(self_prompter, 'goal_patterns')
        assert 'improvement_areas' in self_prompter.goal_patterns
    
    def test_propose_goals_empty_analysis(self, self_prompter):
        """Test goal proposal with empty analysis data"""
        # Mock empty analysis data
        with patch.object(self_prompter, '_gather_analysis_data') as mock_gather:
            mock_gather.return_value = {
                'memory_stats': {},
                'plan_stats': {},
                'intention_stats': {},
                'reflection_data': {},
                'performance_patterns': {}
            }
            
            goals = self_prompter.propose_goals()
            assert isinstance(goals, list)
            # Should still generate some goals from patterns
    
    def test_propose_goals_with_contradictions(self, self_prompter):
        """Test goal proposal when contradictions exist"""
        with patch.object(self_prompter, '_gather_analysis_data') as mock_gather:
            mock_gather.return_value = {
                'memory_stats': {
                    'contradictions': {'count': 10}
                },
                'plan_stats': {},
                'intention_stats': {},
                'reflection_data': {},
                'performance_patterns': {}
            }
            
            goals = self_prompter.propose_goals()
            assert isinstance(goals, list)
            # Should include contradiction resolution goal
            contradiction_goals = [g for g in goals if 'contradiction' in g.lower()]
            assert len(contradiction_goals) > 0
    
    def test_propose_goals_with_failed_plans(self, self_prompter):
        """Test goal proposal when failed plans exist"""
        with patch.object(self_prompter, '_gather_analysis_data') as mock_gather:
            mock_gather.return_value = {
                'memory_stats': {},
                'plan_stats': {'average_success_rate': 0.3},
                'failed_plans': [
                    {'plan_id': 'f1', 'plan_type': 'sequential'},
                    {'plan_id': 'f2', 'plan_type': 'sequential'},
                    {'plan_id': 'f3', 'plan_type': 'parallel'},
                    {'plan_id': 'f4', 'plan_type': 'sequential'}
                ],
                'intention_stats': {},
                'reflection_data': {},
                'performance_patterns': {}
            }
            
            goals = self_prompter.propose_goals()
            assert isinstance(goals, list)
            # Should include improvement goals
            improvement_goals = [g for g in goals if 'improve' in g.lower() or 'success rate' in g.lower()]
            assert len(improvement_goals) > 0
    
    def test_prioritize_goals_empty(self, self_prompter):
        """Test goal prioritization with empty list"""
        prioritized = self_prompter.prioritize_goals([])
        assert prioritized == []
    
    def test_prioritize_goals_basic(self, self_prompter):
        """Test basic goal prioritization"""
        goals = [
            "Explore new techniques for learning",
            "Fix critical system failure immediately", 
            "Optimize performance by 20%",
            "Learn something interesting"
        ]
        
        prioritized = self_prompter.prioritize_goals(goals)
        assert len(prioritized) == len(goals)
        # Critical/urgent goals should be prioritized
        assert "critical" in prioritized[0].lower() or "fix" in prioritized[0].lower()
    
    def test_score_goal_priority(self, self_prompter):
        """Test individual goal priority scoring"""
        high_priority_goal = "Fix critical system failure immediately"
        medium_priority_goal = "Improve response accuracy by 15%"
        low_priority_goal = "Explore interesting topics"
        
        high_score = self_prompter._score_goal_priority(high_priority_goal)
        medium_score = self_prompter._score_goal_priority(medium_priority_goal)
        low_score = self_prompter._score_goal_priority(low_priority_goal)
        
        assert high_score > medium_score > low_score
        assert 0.0 <= low_score <= 1.0
        assert 0.0 <= medium_score <= 1.0
        assert 0.0 <= high_score <= 1.0

class TestCLIIntegration:
    """Test CLI command integration"""
    
    def test_recursive_planning_commands_available(self):
        """Test that recursive planning commands are in available commands"""
        from cortex.cli_commands import AVAILABLE_COMMANDS
        
        expected_commands = [
            "plan_goal", "show_plan", "execute_plan", 
            "why_am_i_doing_this", "self_prompt",
            "list_plans", "delete_plan", "plan_stats", "intention_stats"
        ]
        
        for cmd in expected_commands:
            assert cmd in AVAILABLE_COMMANDS, f"Command {cmd} not in available commands"
    
    @patch('cortex.cli_commands.RECURSIVE_PLANNING_MODE', True)
    def test_plan_goal_command_handler(self):
        """Test plan_goal command handler"""
        from cortex.cli_commands import handle_plan_goal_command
        
        with patch('cortex.cli_commands.RecursivePlanner') as mock_planner_class:
            mock_planner = Mock()
            mock_plan = Mock()
            mock_plan.plan_id = "test-plan"
            mock_plan.plan_type = "sequential"
            mock_plan.confidence = 0.8
            mock_plan.priority = 1
            mock_plan.created_at = datetime.now().isoformat()
            mock_plan.steps = []
            mock_plan.success_criteria = []
            mock_plan.risk_factors = []
            
            mock_planner.plan_goal.return_value = mock_plan
            mock_planner.score_plan.return_value = 0.75
            mock_planner_class.return_value = mock_planner
            
            result = handle_plan_goal_command("Test goal")
            assert result == 'continue'
            mock_planner.plan_goal.assert_called_once_with("Test goal")
    
    @patch('cortex.cli_commands.RECURSIVE_PLANNING_MODE', True)
    def test_self_prompt_command_handler(self):
        """Test self_prompt command handler"""
        from cortex.cli_commands import handle_self_prompt_command
        
        with patch('cortex.cli_commands.SelfPromptGenerator') as mock_prompter_class:
            mock_prompter = Mock()
            mock_prompter.propose_goals.return_value = [
                "Improve memory efficiency",
                "Optimize planning algorithms",
                "Enhance contradiction resolution"
            ]
            mock_prompter_class.return_value = mock_prompter
            
            result = handle_self_prompt_command()
            assert result == 'continue'
            mock_prompter.propose_goals.assert_called_once()

class TestIntegrationScenarios:
    """Test complete integration scenarios"""
    
    @pytest.fixture
    def temp_dbs(self):
        """Create temporary databases for integration testing"""
        with tempfile.NamedTemporaryFile(suffix='_plan.db', delete=False) as plan_f:
            plan_db = plan_f.name
        with tempfile.NamedTemporaryFile(suffix='_intention.db', delete=False) as intention_f:
            intention_db = intention_f.name
        
        yield plan_db, intention_db
        
        os.unlink(plan_db)
        os.unlink(intention_db)
    
    def test_complete_planning_workflow(self, temp_dbs):
        """Test complete workflow: goal -> plan -> execution -> intention tracking"""
        plan_db, intention_db = temp_dbs
        
        # Initialize components
        planner = RecursivePlanner()
        plan_memory = PlanMemory(db_path=plan_db)
        intention_model = IntentionModel(db_path=intention_db)
        
        # Mock the planner's memory connections
        planner._plan_memory = plan_memory
        planner._intention_model = intention_model
        
        # Step 1: Record intention for the goal
        goal_id = "workflow-test-goal"
        intention_id = intention_model.record_intention(
            goal_id=goal_id,
            triggered_by="test_scenario",
            drive="testing",
            importance=0.8,
            reflection_note="Testing complete workflow"
        )
        assert intention_id != ""
        
        # Step 2: Generate plan
        with patch.object(planner, '_generate_plan_with_llm') as mock_llm:
            mock_llm.return_value = json.dumps({
                "goal_text": "Complete workflow test",
                "plan_type": "sequential",
                "steps": [
                    {
                        "subgoal": "Initialize system",
                        "why": "Need setup first",
                        "expected_result": "System ready",
                        "prerequisites": [],
                        "resources_needed": ["system"]
                    },
                    {
                        "subgoal": "Execute main task",
                        "why": "Core functionality",
                        "expected_result": "Task completed",
                        "prerequisites": ["Initialize system"],
                        "resources_needed": ["processor"]
                    }
                ],
                "success_criteria": ["All steps completed"],
                "risk_factors": ["System failure"]
            })
            
            plan = planner.plan_goal("Complete workflow test")
            assert plan is not None
            assert len(plan.steps) == 2
        
        # Step 3: Execute plan
        results = planner.execute_plan(plan)
        assert "plan_id" in results
        assert "completion_percentage" in results
        
        # Step 4: Verify intention chain
        trace = intention_model.trace_why(goal_id)
        assert len(trace) >= 1
        assert trace[0]['goal_id'] == goal_id
        
        # Step 5: Check stored data
        retrieved_plan = plan_memory.get_plan_by_id(plan.plan_id)
        assert retrieved_plan is not None
        
        plan_stats = plan_memory.get_plan_statistics()
        assert plan_stats['total_plans'] >= 1
        
        intention_stats = intention_model.get_intention_statistics()
        assert intention_stats['total_intentions'] >= 1
    
    def test_self_directed_improvement_cycle(self, temp_dbs):
        """Test self-directed improvement: analysis -> goals -> plans -> execution"""
        plan_db, intention_db = temp_dbs
        
        # Initialize components
        plan_memory = PlanMemory(db_path=plan_db)
        intention_model = IntentionModel(db_path=intention_db)
        self_prompter = SelfPromptGenerator()
        
        # Mock external systems
        self_prompter._plan_memory = plan_memory
        self_prompter._intention_model = intention_model
        self_prompter._memory_system = Mock()
        self_prompter._reflection_system = Mock()
        
        # Create some existing failed plans to trigger self-improvement
        failed_plan = Plan("failed-1", "Previous failed goal", [
            PlanStep("fs1", "Failed step", "Bad reasoning", "No result", [])
        ])
        plan_memory.store_plan(failed_plan)
        
        # Record failure outcome
        failure_outcome = {
            "overall_success": False,
            "completion_percentage": 20.0,
            "execution_start": datetime.now().isoformat(),
            "execution_end": datetime.now().isoformat(),
            "steps_executed": [],
            "steps_failed": [{"step_id": "fs1", "subgoal": "Failed step", "reason": "Poor planning"}]
        }
        plan_memory.record_plan_outcome("failed-1", failure_outcome)
        
        # Mock analysis data to trigger goal generation
        with patch.object(self_prompter, '_gather_analysis_data') as mock_gather:
            mock_gather.return_value = {
                'memory_stats': {},
                'plan_stats': {'average_success_rate': 0.2},
                'failed_plans': [{'plan_id': 'failed-1', 'plan_type': 'sequential'}],
                'intention_stats': {'fulfillment_rate': 0.3},
                'reflection_data': {},
                'performance_patterns': {}
            }
            
            # Generate self-directed goals
            goals = self_prompter.propose_goals()
            assert len(goals) > 0
            
            # Prioritize goals
            prioritized_goals = self_prompter.prioritize_goals(goals)
            assert len(prioritized_goals) == len(goals)
            
            # Verify improvement-focused goals were generated
            improvement_found = any(
                'improve' in goal.lower() or 'success rate' in goal.lower() 
                for goal in prioritized_goals
            )
            assert improvement_found, "No improvement goals generated despite poor performance"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])