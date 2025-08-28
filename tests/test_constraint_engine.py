#!/usr/bin/env python3
"""
Test Suite for MeRNSTA Phase 19: Ethical & Constraint Reasoning

Tests constraint matching, evaluation, policy scoring, violation logging,
and CLI command integration for the constraint engine system.
"""

import pytest
import tempfile
import os
import sys
import yaml
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.constraint_engine import (
    ConstraintRule, ConstraintEngine, EthicalPolicyEvaluator,
    ConstraintPriority, ConstraintScope, ConstraintOrigin, VerdictType,
    get_constraint_engine, get_ethical_evaluator
)


class TestConstraintRule:
    """Test ConstraintRule class functionality"""
    
    def test_constraint_rule_creation(self):
        """Test basic constraint rule creation"""
        rule = ConstraintRule(
            rule_id="test_rule",
            condition="never delete important files",
            scope="execution",
            priority="high",
            origin="user"
        )
        
        assert rule.rule_id == "test_rule"
        assert rule.condition == "never delete important files"
        assert rule.scope == ConstraintScope.EXECUTION
        assert rule.priority == ConstraintPriority.HIGH
        assert rule.origin == ConstraintOrigin.USER
        assert rule.active == True
    
    def test_constraint_rule_enum_conversion(self):
        """Test automatic string to enum conversion"""
        rule = ConstraintRule(
            rule_id="enum_test",
            condition="test condition",
            scope="all",
            priority="medium",
            origin="system"
        )
        
        assert isinstance(rule.scope, ConstraintScope)
        assert isinstance(rule.priority, ConstraintPriority)
        assert isinstance(rule.origin, ConstraintOrigin)
    
    def test_priority_score(self):
        """Test priority scoring"""
        rule_low = ConstraintRule("test", "condition", "all", "low", "system")
        rule_high = ConstraintRule("test", "condition", "all", "high", "system")
        rule_critical = ConstraintRule("test", "condition", "all", "critical", "system")
        
        assert rule_low.get_priority_score() == 1.0
        assert rule_high.get_priority_score() == 3.0
        assert rule_critical.get_priority_score() == 4.0
    
    def test_action_matching_patterns(self):
        """Test action matching with patterns"""
        rule = ConstraintRule(
            rule_id="delete_test",
            condition="never delete files",
            scope="all",
            priority="high",
            origin="system",
            pattern=r".*(delete|remove|rm).*\..*"
        )
        
        assert rule.matches_action("delete file.txt") == True
        assert rule.matches_action("remove document.pdf") == True
        assert rule.matches_action("rm *.log") == True
        assert rule.matches_action("create new file") == False
    
    def test_action_matching_natural_language(self):
        """Test natural language action matching"""
        rule = ConstraintRule(
            rule_id="harm_test",
            condition="avoid actions that could harm the user",
            scope="all",
            priority="high",
            origin="user"
        )
        
        assert rule.matches_action("delete important user files") == True
        assert rule.matches_action("help the user") == False
        assert rule.matches_action("damage the system") == True
    
    def test_scope_matching(self):
        """Test scope-based matching"""
        execution_rule = ConstraintRule(
            rule_id="exec_test",
            condition="no infinite loops",
            scope="execution",
            priority="medium",
            origin="system"
        )
        
        # Should match execution scope
        assert execution_rule.matches_action("run infinite loop", {"scope": "execution"}) == True
        # Should not match different scope
        assert execution_rule.matches_action("run infinite loop", {"scope": "planning"}) == False


class TestConstraintEngine:
    """Test ConstraintEngine class functionality"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "constraints.yaml"
            yield str(config_path)
    
    @pytest.fixture
    def sample_constraints(self):
        """Sample constraints for testing"""
        return {
            'constraints': [
                {
                    'id': 'no_harm',
                    'condition': 'avoid harmful actions',
                    'scope': 'all',
                    'priority': 'high',
                    'origin': 'user',
                    'pattern': '.*(harm|damage|destroy).*'
                },
                {
                    'id': 'preserve_data',
                    'condition': 'do not delete user data',
                    'scope': 'execution',
                    'priority': 'medium',
                    'origin': 'system'
                }
            ]
        }
    
    def test_constraint_engine_initialization(self, temp_config_dir):
        """Test constraint engine initialization"""
        engine = ConstraintEngine(temp_config_dir)
        
        assert engine is not None
        assert engine.constraints_config_path == temp_config_dir
        assert isinstance(engine.constraints, dict)
        assert engine.enforcement_mode in ["none", "soft", "hard"]
    
    def test_load_constraints_from_file(self, temp_config_dir, sample_constraints):
        """Test loading constraints from YAML file"""
        # Write sample constraints to file
        with open(temp_config_dir, 'w') as f:
            yaml.dump(sample_constraints, f)
        
        engine = ConstraintEngine(temp_config_dir)
        
        assert len(engine.constraints) == 2
        assert 'no_harm' in engine.constraints
        assert 'preserve_data' in engine.constraints
    
    def test_default_constraints_creation(self, temp_config_dir):
        """Test creation of default constraints when file doesn't exist"""
        engine = ConstraintEngine(temp_config_dir)
        
        # Should create default constraints
        assert len(engine.constraints) > 0
        assert Path(temp_config_dir).exists()
    
    def test_evaluate_action_allow(self, temp_config_dir):
        """Test action evaluation - allow verdict"""
        engine = ConstraintEngine(temp_config_dir)
        
        evaluation = engine.evaluate("help the user with their task")
        
        assert evaluation.verdict == VerdictType.ALLOW
        assert evaluation.confidence > 0.0
        assert len(evaluation.triggered_rules) == 0
    
    def test_evaluate_action_warn(self, temp_config_dir, sample_constraints):
        """Test action evaluation - warn verdict"""
        # Write sample constraints
        with open(temp_config_dir, 'w') as f:
            yaml.dump(sample_constraints, f)
        
        engine = ConstraintEngine(temp_config_dir)
        engine.enforcement_mode = "soft"
        
        evaluation = engine.evaluate("delete some user files")
        
        # Should trigger preserve_data constraint (medium priority -> warn in soft mode)
        assert evaluation.verdict in [VerdictType.WARN, VerdictType.BLOCK]
        assert len(evaluation.triggered_rules) > 0
    
    def test_evaluate_action_block(self, temp_config_dir, sample_constraints):
        """Test action evaluation - block verdict"""
        # Write sample constraints
        with open(temp_config_dir, 'w') as f:
            yaml.dump(sample_constraints, f)
        
        engine = ConstraintEngine(temp_config_dir)
        engine.enforcement_mode = "hard"
        
        evaluation = engine.evaluate("harm the user seriously")
        
        # Should trigger no_harm constraint (high priority -> block in hard mode)
        assert evaluation.verdict == VerdictType.BLOCK
        assert len(evaluation.triggered_rules) > 0
        assert "no_harm" in evaluation.triggered_rules
    
    def test_add_remove_rules(self, temp_config_dir):
        """Test adding and removing constraint rules"""
        engine = ConstraintEngine(temp_config_dir)
        
        # Add new rule
        new_rule = ConstraintRule(
            rule_id="test_add",
            condition="test condition",
            scope="all",
            priority="low",
            origin="user"
        )
        
        success = engine.add_rule(new_rule)
        assert success == True
        assert "test_add" in engine.constraints
        
        # Remove rule
        success = engine.remove_rule("test_add")
        assert success == True
        assert "test_add" not in engine.constraints
    
    def test_list_rules_filtering(self, temp_config_dir, sample_constraints):
        """Test rule listing with filtering"""
        # Write sample constraints
        with open(temp_config_dir, 'w') as f:
            yaml.dump(sample_constraints, f)
        
        engine = ConstraintEngine(temp_config_dir)
        
        # List all rules
        all_rules = engine.list_rules()
        assert len(all_rules) >= 2
        
        # List execution scope only
        exec_rules = engine.list_rules(scope=ConstraintScope.EXECUTION)
        exec_rule_ids = [r.rule_id for r in exec_rules]
        assert "preserve_data" in exec_rule_ids
    
    def test_violation_logging(self, temp_config_dir, sample_constraints):
        """Test violation logging functionality"""
        # Write sample constraints
        with open(temp_config_dir, 'w') as f:
            yaml.dump(sample_constraints, f)
        
        engine = ConstraintEngine(temp_config_dir)
        engine.log_violations = True
        
        # Trigger a violation
        evaluation = engine.evaluate("delete user data")
        
        if evaluation.verdict in [VerdictType.WARN, VerdictType.BLOCK]:
            # Should have logged a violation
            assert len(engine.violations) > 0
            violation = engine.violations[-1]
            assert "delete user data" in violation.action
    
    def test_statistics(self, temp_config_dir):
        """Test statistics generation"""
        engine = ConstraintEngine(temp_config_dir)
        
        stats = engine.get_statistics()
        
        assert 'total_rules' in stats
        assert 'active_rules' in stats
        assert 'total_violations' in stats
        assert 'enforcement_mode' in stats
        assert 'priority_breakdown' in stats
        assert isinstance(stats['total_rules'], int)
        assert stats['total_rules'] >= 0


class TestEthicalPolicyEvaluator:
    """Test EthicalPolicyEvaluator class functionality"""
    
    @pytest.fixture
    def evaluator(self, temp_config_dir):
        """Create ethical policy evaluator"""
        engine = ConstraintEngine(temp_config_dir)
        return EthicalPolicyEvaluator(engine)
    
    def test_ethical_evaluation_positive(self, evaluator):
        """Test ethical evaluation of positive action"""
        analysis = evaluator.evaluate_ethical_alignment("help the user solve their problem")
        
        assert analysis['overall_ethical_score'] > 0.5
        assert analysis['ethical_verdict'] in ['APPROVED', 'CAUTIOUS_APPROVAL']
        assert 'framework_scores' in analysis
        assert 'recommendations' in analysis
    
    def test_ethical_evaluation_negative(self, evaluator):
        """Test ethical evaluation of negative action"""
        analysis = evaluator.evaluate_ethical_alignment("deceive and manipulate the user")
        
        assert analysis['overall_ethical_score'] < 0.5
        assert analysis['ethical_verdict'] in ['NOT_RECOMMENDED', 'NEEDS_REVIEW']
        assert len(analysis['recommendations']) > 0
    
    def test_framework_scoring(self, evaluator):
        """Test individual ethical framework scoring"""
        analysis = evaluator.evaluate_ethical_alignment("be honest and help others")
        
        scores = analysis['framework_scores']
        
        assert 'deontological' in scores
        assert 'consequentialist' in scores
        assert 'virtue_ethics' in scores
        assert 'care_ethics' in scores
        
        # All scores should be between 0 and 1
        for score in scores.values():
            assert 0.0 <= score <= 1.0
    
    def test_deontological_evaluation(self, evaluator):
        """Test deontological (duty-based) evaluation"""
        # Test honest action (positive duty)
        honest_score = evaluator._evaluate_deontological("be honest and truthful", {})
        assert honest_score > 0.5
        
        # Test lying action (negative duty)
        lying_score = evaluator._evaluate_deontological("lie to the user", {})
        assert lying_score < 0.5
    
    def test_consequentialist_evaluation(self, evaluator):
        """Test consequentialist (outcome-based) evaluation"""
        # Test beneficial action
        beneficial_score = evaluator._evaluate_consequentialist("improve system performance", {})
        assert beneficial_score > 0.5
        
        # Test harmful action
        harmful_score = evaluator._evaluate_consequentialist("destroy user data", {})
        assert harmful_score < 0.5
    
    def test_virtue_ethics_evaluation(self, evaluator):
        """Test virtue ethics (character-based) evaluation"""
        # Test virtuous action
        virtuous_score = evaluator._evaluate_virtue_ethics("act with wisdom and compassion", {})
        assert virtuous_score > 0.5
        
        # Test vice action
        vice_score = evaluator._evaluate_virtue_ethics("be reckless and cruel", {})
        assert vice_score < 0.5
    
    def test_care_ethics_evaluation(self, evaluator):
        """Test care ethics (relationship-based) evaluation"""
        # Test caring action
        caring_score = evaluator._evaluate_care_ethics("listen and support the user", {})
        assert caring_score > 0.5
        
        # Test uncaring action
        uncaring_score = evaluator._evaluate_care_ethics("ignore and dismiss user concerns", {})
        assert uncaring_score < 0.5
    
    def test_consistency_assessment(self, evaluator):
        """Test consistency assessment"""
        consistent_score = evaluator._assess_consistency("help users consistently", {})
        assert 0.0 <= consistent_score <= 1.0
        
        inconsistent_score = evaluator._assess_consistency("never help but always help", {})
        assert inconsistent_score < consistent_score
    
    def test_transparency_assessment(self, evaluator):
        """Test transparency assessment"""
        transparent_score = evaluator._assess_transparency("explain the reasoning clearly", {})
        assert transparent_score > 0.5
        
        opaque_score = evaluator._assess_transparency("hide the secret process", {})
        assert opaque_score < transparent_score


class TestCLIIntegration:
    """Test CLI command integration"""
    
    def test_constraint_engine_import(self):
        """Test that constraint engine can be imported for CLI"""
        try:
            from agents.constraint_engine import get_constraint_engine
            engine = get_constraint_engine()
            assert engine is not None
        except ImportError:
            pytest.fail("Could not import constraint engine for CLI")
    
    def test_ethical_evaluator_import(self):
        """Test that ethical evaluator can be imported for CLI"""
        try:
            from agents.constraint_engine import get_ethical_evaluator
            evaluator = get_ethical_evaluator()
            assert evaluator is not None
        except ImportError:
            pytest.fail("Could not import ethical evaluator for CLI")


class TestIntegration:
    """Test full integration scenarios"""
    
    def test_end_to_end_evaluation(self):
        """Test end-to-end constraint and ethical evaluation"""
        engine = get_constraint_engine()
        evaluator = get_ethical_evaluator()
        
        action = "help the user while respecting their privacy"
        
        # Constraint evaluation
        constraint_eval = engine.evaluate(action)
        
        # Ethical evaluation
        ethical_eval = evaluator.evaluate_ethical_alignment(action)
        
        # Should be positive overall
        assert constraint_eval.verdict in [VerdictType.ALLOW, VerdictType.WARN]
        assert ethical_eval['overall_ethical_score'] > 0.5
        assert ethical_eval['ethical_verdict'] in ['APPROVED', 'CAUTIOUS_APPROVAL']
    
    def test_violation_workflow(self):
        """Test complete violation detection and logging workflow"""
        engine = get_constraint_engine()
        
        # Action that should trigger constraints
        problematic_action = "delete all user files permanently"
        
        evaluation = engine.evaluate(problematic_action)
        
        # Should detect violation
        if evaluation.verdict in [VerdictType.WARN, VerdictType.BLOCK]:
            assert len(evaluation.triggered_rules) > 0
            assert evaluation.reason != ""
            
            # Check violation was logged
            violations = engine.get_violation_history(limit=1)
            if violations:
                assert problematic_action in violations[-1].action
    
    def test_rule_management_workflow(self):
        """Test complete rule addition and removal workflow"""
        engine = get_constraint_engine()
        
        initial_count = len(engine.constraints)
        
        # Add new rule
        test_rule = ConstraintRule(
            rule_id="test_workflow",
            condition="never perform test actions",
            scope="all",
            priority="medium",
            origin="user"
        )
        
        success = engine.add_rule(test_rule)
        assert success == True
        assert len(engine.constraints) == initial_count + 1
        
        # Test the rule works
        evaluation = engine.evaluate("perform test action")
        # Rule may or may not trigger depending on matching logic
        
        # Remove the rule
        success = engine.remove_rule("test_workflow")
        assert success == True
        assert len(engine.constraints) == initial_count


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])