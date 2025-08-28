#!/usr/bin/env python3
"""
Comprehensive tests for Agent Contract System (Phase 26)

Tests contract functionality including:
- Contract creation and serialization
- Task alignment scoring
- Performance feedback updates
- Specialization drift tracking
- Contract loading and saving
"""

import unittest
import tempfile
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.agent_contract import (
    AgentContract, 
    create_default_contracts, 
    load_or_create_contract,
    score_all_agents_for_task
)


class TestAgentContract(unittest.TestCase):
    """Test basic AgentContract functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_contract = AgentContract(
            agent_name="test_agent",
            purpose="Test agent for unit testing",
            capabilities=["testing", "validation", "debugging"],
            preferred_tasks=["unit_tests", "integration_tests", "bug_fixing"],
            weaknesses=["performance_optimization", "user_interfaces"],
            confidence_vector={
                "testing": 0.9,
                "debugging": 0.8,
                "analysis": 0.7,
                "planning": 0.5,
                "creativity": 0.3
            }
        )
    
    def test_contract_creation(self):
        """Test basic contract creation."""
        self.assertEqual(self.test_contract.agent_name, "test_agent")
        self.assertEqual(self.test_contract.purpose, "Test agent for unit testing")
        self.assertIn("testing", self.test_contract.capabilities)
        self.assertIn("unit_tests", self.test_contract.preferred_tasks)
        self.assertIn("performance_optimization", self.test_contract.weaknesses)
        self.assertEqual(self.test_contract.confidence_vector["testing"], 0.9)
    
    def test_default_confidence_vector(self):
        """Test default confidence vector generation."""
        minimal_contract = AgentContract(
            agent_name="minimal_agent",
            purpose="Minimal test agent"
        )
        
        # Should have default confidence vector
        self.assertIsNotNone(minimal_contract.confidence_vector)
        self.assertIn("planning", minimal_contract.confidence_vector)
        self.assertIn("reasoning", minimal_contract.confidence_vector)
        
        # Values should be reasonable defaults
        for skill, confidence in minimal_contract.confidence_vector.items():
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
    
    def test_serialization(self):
        """Test contract serialization and deserialization."""
        # Convert to dict
        contract_dict = self.test_contract.to_dict()
        
        # Verify essential fields
        self.assertEqual(contract_dict["agent_name"], "test_agent")
        self.assertEqual(contract_dict["purpose"], "Test agent for unit testing")
        self.assertIsInstance(contract_dict["capabilities"], list)
        self.assertIsInstance(contract_dict["confidence_vector"], dict)
        self.assertIsInstance(contract_dict["last_updated"], str)
        
        # Recreate from dict
        recreated_contract = AgentContract.from_dict(contract_dict)
        
        # Verify recreation
        self.assertEqual(recreated_contract.agent_name, self.test_contract.agent_name)
        self.assertEqual(recreated_contract.purpose, self.test_contract.purpose)
        self.assertEqual(recreated_contract.capabilities, self.test_contract.capabilities)
        self.assertEqual(recreated_contract.confidence_vector, self.test_contract.confidence_vector)
    
    def test_file_operations(self):
        """Test saving and loading contracts from files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_contract.json"
            
            # Save contract
            self.test_contract.save_to_file(test_file)
            self.assertTrue(test_file.exists())
            
            # Load contract
            loaded_contract = AgentContract.load_from_file(test_file)
            self.assertIsNotNone(loaded_contract)
            self.assertEqual(loaded_contract.agent_name, "test_agent")
            self.assertEqual(loaded_contract.purpose, self.test_contract.purpose)
            self.assertEqual(loaded_contract.capabilities, self.test_contract.capabilities)
    
    def test_contract_summary(self):
        """Test contract summary generation."""
        summary = self.test_contract.get_summary()
        
        # Verify summary fields
        self.assertIn("agent_name", summary)
        self.assertIn("purpose", summary)
        self.assertIn("version", summary)
        self.assertIn("top_capabilities", summary)
        self.assertIn("total_capabilities", summary)
        
        # Verify top capabilities are sorted by confidence
        top_caps = summary["top_capabilities"]
        self.assertGreater(len(top_caps), 0)
        
        # Should be sorted in descending order of confidence
        if len(top_caps) > 1:
            self.assertGreaterEqual(top_caps[0][1], top_caps[1][1])


class TestTaskAlignmentScoring(unittest.TestCase):
    """Test task alignment scoring functionality."""
    
    def setUp(self):
        """Set up test agents with different specializations."""
        self.planner_contract = AgentContract(
            agent_name="planner",
            purpose="Break down complex tasks into actionable steps",
            capabilities=["task_decomposition", "step_sequencing", "planning"],
            preferred_tasks=["planning", "organization", "project_breakdown"],
            confidence_vector={
                "planning": 0.9,
                "reasoning": 0.8,
                "execution": 0.6,
                "analysis": 0.7
            }
        )
        
        self.critic_contract = AgentContract(
            agent_name="critic",
            purpose="Identify flaws and potential issues",
            capabilities=["critical_analysis", "flaw_detection", "risk_assessment"],
            preferred_tasks=["analysis", "review", "evaluation"],
            confidence_vector={
                "analysis": 0.9,
                "reasoning": 0.8,
                "planning": 0.4,
                "creativity": 0.3
            }
        )
        
        self.debugger_contract = AgentContract(
            agent_name="debugger",
            purpose="Find and fix bugs in code",
            capabilities=["debugging", "error_detection", "code_analysis"],
            preferred_tasks=["debugging", "troubleshooting", "error_fixing"],
            weaknesses=["creative_design", "planning"],
            confidence_vector={
                "debugging": 0.95,
                "analysis": 0.8,
                "execution": 0.7,
                "creativity": 0.2
            }
        )
    
    def test_basic_task_scoring(self):
        """Test basic task alignment scoring."""
        # Test planning task
        planning_task = "Create a detailed project plan for software development"
        planner_score = self.planner_contract.score_alignment(planning_task)
        critic_score = self.critic_contract.score_alignment(planning_task)
        
        # Planner should score higher for planning tasks
        self.assertGreater(planner_score, critic_score)
        self.assertGreater(planner_score, 0.5)  # Should be a good fit
        
        # Test analysis task
        analysis_task = "Analyze system performance and identify bottlenecks"
        planner_analysis_score = self.planner_contract.score_alignment(analysis_task)
        critic_analysis_score = self.critic_contract.score_alignment(analysis_task)
        
        # Critic should score higher for analysis tasks
        self.assertGreater(critic_analysis_score, planner_analysis_score)
        
        # Test debugging task
        debug_task = "Fix memory leak in application causing crashes"
        debugger_score = self.debugger_contract.score_alignment(debug_task)
        planner_debug_score = self.planner_contract.score_alignment(debug_task)
        
        # Debugger should score highest for debugging tasks
        self.assertGreater(debugger_score, planner_debug_score)
        self.assertGreater(debugger_score, 0.6)  # Should be good fit
    
    def test_task_with_context(self):
        """Test task scoring with additional context."""
        task = "Optimize database queries for better performance"
        context = {
            'type': 'optimization',
            'urgency': 0.8,
            'complexity': 0.7,
            'keywords': ['optimize', 'database', 'performance']
        }
        
        # Test with context
        score_with_context = self.debugger_contract.score_alignment({
            'description': task,
            **context
        })
        
        # Test without context
        score_without_context = self.debugger_contract.score_alignment(task)
        
        # Both should return valid scores
        self.assertGreaterEqual(score_with_context, 0.0)
        self.assertLessEqual(score_with_context, 1.0)
        self.assertGreaterEqual(score_without_context, 0.0)
        self.assertLessEqual(score_without_context, 1.0)
    
    def test_weakness_penalty(self):
        """Test that known weaknesses reduce alignment scores."""
        # Create a task that hits debugger's weakness
        creative_task = "Design an innovative user interface for mobile app"
        
        debugger_score = self.debugger_contract.score_alignment(creative_task)
        
        # Should have lower score due to creativity weakness
        self.assertLess(debugger_score, 0.6)
    
    def test_score_bounds(self):
        """Test that scores are always within valid bounds."""
        test_tasks = [
            "Simple task",
            "Complex comprehensive system analysis with multiple interdependent components",
            "Debug critical production issue",
            "Plan strategic roadmap",
            "Creative brainstorming session",
            ""  # Edge case: empty task
        ]
        
        contracts = [self.planner_contract, self.critic_contract, self.debugger_contract]
        
        for task in test_tasks:
            for contract in contracts:
                score = contract.score_alignment(task)
                self.assertGreaterEqual(score, 0.0, f"Score below 0 for {contract.agent_name} on '{task}'")
                self.assertLessEqual(score, 1.0, f"Score above 1 for {contract.agent_name} on '{task}'")


class TestPerformanceFeedback(unittest.TestCase):
    """Test performance feedback and learning functionality."""
    
    def setUp(self):
        """Set up test contract for feedback testing."""
        self.test_contract = AgentContract(
            agent_name="learning_agent",
            purpose="Agent that learns from feedback",
            capabilities=["learning", "adaptation"],
            confidence_vector={
                "planning": 0.6,
                "debugging": 0.5,
                "analysis": 0.7
            }
        )
    
    def test_positive_feedback_update(self):
        """Test that positive feedback improves confidence."""
        initial_planning_confidence = self.test_contract.confidence_vector["planning"]
        
        # Positive feedback for planning task
        feedback = {
            "task_type": "planning",
            "success": True,
            "performance_score": 0.9,
            "quality_rating": 0.8,
            "notes": "Excellent task breakdown"
        }
        
        self.test_contract.update_from_performance_feedback(feedback)
        
        # Check that planning confidence increased
        new_planning_confidence = self.test_contract.confidence_vector["planning"]
        self.assertGreater(new_planning_confidence, initial_planning_confidence)
        
        # Check performance history
        self.assertEqual(len(self.test_contract.performance_history), 1)
        self.assertTrue(self.test_contract.performance_history[0]["success"])
    
    def test_negative_feedback_update(self):
        """Test that negative feedback decreases confidence."""
        initial_debugging_confidence = self.test_contract.confidence_vector["debugging"]
        
        # Negative feedback for debugging task
        feedback = {
            "task_type": "debugging",
            "success": False,
            "performance_score": 0.2,
            "quality_rating": 0.3,
            "notes": "Failed to identify root cause"
        }
        
        self.test_contract.update_from_performance_feedback(feedback)
        
        # Check that debugging confidence decreased
        new_debugging_confidence = self.test_contract.confidence_vector["debugging"]
        self.assertLess(new_debugging_confidence, initial_debugging_confidence)
        
        # Check performance history
        self.assertEqual(len(self.test_contract.performance_history), 1)
        self.assertFalse(self.test_contract.performance_history[0]["success"])
    
    def test_specialization_drift_tracking(self):
        """Test specialization drift tracking over time."""
        # Add multiple feedback entries for same task type
        feedback_entries = [
            {"task_type": "analysis", "success": True, "performance_score": 0.6},
            {"task_type": "analysis", "success": True, "performance_score": 0.7},
            {"task_type": "analysis", "success": True, "performance_score": 0.8},
            {"task_type": "analysis", "success": True, "performance_score": 0.9}
        ]
        
        for feedback in feedback_entries:
            self.test_contract.update_from_performance_feedback(feedback)
        
        # Check specialization drift tracking
        self.assertIn("analysis", self.test_contract.specialization_drift)
        self.assertEqual(len(self.test_contract.specialization_drift["analysis"]), 4)
        
        # Get trend (should be positive - improving)
        trend = self.test_contract.get_specialization_trend("analysis")
        self.assertIsNotNone(trend)
        self.assertGreater(trend, 0)  # Improving trend
    
    def test_performance_history_limits(self):
        """Test that performance history is limited to prevent memory bloat."""
        # Add many feedback entries
        for i in range(150):  # More than the 100 limit
            feedback = {
                "task_type": "test",
                "success": True,
                "performance_score": 0.5,
                "notes": f"Task {i}"
            }
            self.test_contract.update_from_performance_feedback(feedback)
        
        # Should be limited to 100 entries
        self.assertLessEqual(len(self.test_contract.performance_history), 100)


class TestDefaultContracts(unittest.TestCase):
    """Test default contract creation and loading functionality."""
    
    def test_create_default_contracts(self):
        """Test creation of default contracts for known agents."""
        default_contracts = create_default_contracts()
        
        # Should include common agent types
        expected_agents = ["planner", "critic", "debater", "reflector", "architect_analyzer"]
        
        for agent_name in expected_agents:
            self.assertIn(agent_name, default_contracts)
            contract = default_contracts[agent_name]
            self.assertIsInstance(contract, AgentContract)
            self.assertEqual(contract.agent_name, agent_name)
            self.assertIsNotNone(contract.purpose)
            self.assertGreater(len(contract.capabilities), 0)
    
    def test_load_or_create_contract(self):
        """Test loading existing contracts or creating new ones."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test creating new contract
            contract1 = load_or_create_contract("test_agent", temp_dir)
            self.assertIsNotNone(contract1)
            self.assertEqual(contract1.agent_name, "test_agent")
            
            # Test loading existing contract
            contract2 = load_or_create_contract("test_agent", temp_dir)
            self.assertIsNotNone(contract2)
            self.assertEqual(contract2.agent_name, "test_agent")
            
            # Should be the same contract loaded from file
            self.assertEqual(contract1.purpose, contract2.purpose)
    
    def test_score_all_agents_function(self):
        """Test the score_all_agents_for_task function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test contracts
            contracts = create_default_contracts()
            for agent_name, contract in contracts.items():
                contract.save_to_file(Path(temp_dir) / f"{agent_name}.json")
            
            # Score a task
            task = "Plan a software development project"
            agent_scores = score_all_agents_for_task(task, temp_dir)
            
            self.assertGreater(len(agent_scores), 0)
            
            # Should be sorted by score (highest first)
            for i in range(len(agent_scores) - 1):
                self.assertGreaterEqual(
                    agent_scores[i]["alignment_score"],
                    agent_scores[i + 1]["alignment_score"]
                )
            
            # Each result should have required fields
            for result in agent_scores:
                self.assertIn("agent_name", result)
                self.assertIn("alignment_score", result)
                self.assertIn("purpose", result)
                self.assertIn("confidence_vector", result)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_task_scoring(self):
        """Test scoring with empty or invalid tasks."""
        contract = AgentContract(
            agent_name="test_agent",
            purpose="Test agent"
        )
        
        # Empty string
        score1 = contract.score_alignment("")
        self.assertGreaterEqual(score1, 0.0)
        self.assertLessEqual(score1, 1.0)
        
        # Very long task
        long_task = "analyze " * 1000  # Very repetitive long task
        score2 = contract.score_alignment(long_task)
        self.assertGreaterEqual(score2, 0.0)
        self.assertLessEqual(score2, 1.0)
    
    def test_invalid_feedback(self):
        """Test handling of invalid performance feedback."""
        contract = AgentContract(
            agent_name="test_agent",
            purpose="Test agent"
        )
        
        # Empty feedback
        try:
            contract.update_from_performance_feedback({})
            # Should not crash
        except Exception as e:
            self.fail(f"Empty feedback should not cause crash: {e}")
        
        # Invalid feedback types
        try:
            contract.update_from_performance_feedback({
                "task_type": "test",
                "success": "maybe",  # Invalid boolean
                "performance_score": "high"  # Invalid number
            })
            # Should not crash
        except Exception as e:
            self.fail(f"Invalid feedback should not cause crash: {e}")
    
    def test_file_operation_errors(self):
        """Test handling of file operation errors."""
        contract = AgentContract(
            agent_name="test_agent",
            purpose="Test agent"
        )
        
        # Try to save to invalid path
        try:
            contract.save_to_file("/invalid/path/contract.json")
            # Should handle gracefully
        except Exception:
            pass  # Expected to fail, but shouldn't crash the test
        
        # Try to load non-existent file
        loaded = AgentContract.load_from_file("/non/existent/file.json")
        self.assertIsNone(loaded)
    
    def test_contract_age_calculation(self):
        """Test contract age and staleness detection."""
        # Create old contract
        old_contract = AgentContract(
            agent_name="old_agent",
            purpose="Old test agent",
            last_updated=datetime.now() - timedelta(days=45)
        )
        
        # Create recent contract
        recent_contract = AgentContract(
            agent_name="recent_agent",
            purpose="Recent test agent"
        )
        
        # Age calculation should work
        old_age = datetime.now() - old_contract.last_updated
        recent_age = datetime.now() - recent_contract.last_updated
        
        self.assertGreater(old_age.days, 30)
        self.assertLess(recent_age.days, 1)


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)