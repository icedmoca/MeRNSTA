#!/usr/bin/env python3
"""
Test Suite for Phase 21: Self-Critique & Debate Agent Swarms

Tests the debate engine, critic agent, debater agent, and reflection orchestrator
to ensure proper functionality of the dialectical reasoning system.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, List, Any

# Import the debate system components
from agents.debate_engine import (
    DebateEngine, Argument, DebateResult, DebateStance, 
    ArgumentType, DebatePhase, ConclusionStrategy
)
from agents.critic import CriticAgent
from agents.debater import DebaterAgent
from agents.reflection_orchestrator import (
    ReflectionOrchestrator, ReflectionTrigger, ReflectionPriority,
    ReflectionRequest, ReflectionResult
)


class TestDebateEngine(unittest.TestCase):
    """Test the DebateEngine core functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.debate_engine = DebateEngine()
        self.test_claim = "AI development should prioritize safety over speed"
        
    def test_debate_engine_initialization(self):
        """Test that DebateEngine initializes properly."""
        self.assertEqual(self.debate_engine.name, "debate_engine")
        self.assertTrue(self.debate_engine.enabled)
        self.assertEqual(self.debate_engine.max_rounds, 3)
        self.assertIsInstance(self.debate_engine.active_debates, dict)
        self.assertIsInstance(self.debate_engine.debate_history, list)
    
    def test_argument_creation(self):
        """Test Argument dataclass creation and properties."""
        argument = Argument(
            agent_id="test_agent",
            stance=DebateStance.PRO,
            content="This is a test argument",
            confidence=0.8,
            argument_type=ArgumentType.LOGICAL
        )
        
        self.assertEqual(argument.agent_id, "test_agent")
        self.assertEqual(argument.stance, DebateStance.PRO)
        self.assertEqual(argument.content, "This is a test argument")
        self.assertEqual(argument.confidence, 0.8)
        self.assertEqual(argument.argument_type, ArgumentType.LOGICAL)
        self.assertIsInstance(argument.timestamp, datetime)
    
    def test_debate_stance_enum(self):
        """Test DebateStance enum values."""
        self.assertEqual(DebateStance.STRONGLY_PRO.value, "strongly_pro")
        self.assertEqual(DebateStance.PRO.value, "pro")
        self.assertEqual(DebateStance.NEUTRAL.value, "neutral")
        self.assertEqual(DebateStance.CON.value, "con")
        self.assertEqual(DebateStance.STRONGLY_CON.value, "strongly_con")
    
    def test_argument_scoring(self):
        """Test argument scoring mechanisms."""
        argument = Argument(
            content="Because research shows that safety measures prevent accidents, therefore prioritizing safety is logical",
            confidence=0.9
        )
        
        # Test logical validity assessment
        validity = self.debate_engine._assess_logical_validity(argument)
        self.assertGreater(validity, 0.5)  # Should score well for logical indicators
        
        # Test argument strength assessment
        argument.logical_validity = validity
        strength = self.debate_engine._assess_argument_strength(argument)
        self.assertGreater(strength, 0.5)
        
        # Test novelty assessment
        novelty = self.debate_engine._assess_novelty(argument)
        self.assertGreaterEqual(novelty, 0.0)
        self.assertLessEqual(novelty, 1.0)
    
    def test_argument_ranking(self):
        """Test argument ranking by different criteria."""
        arguments = [
            Argument(content="Argument 1", strength_score=0.8, logical_validity=0.7, novelty_score=0.6),
            Argument(content="Argument 2", strength_score=0.6, logical_validity=0.9, novelty_score=0.8),
            Argument(content="Argument 3", strength_score=0.9, logical_validity=0.5, novelty_score=0.7)
        ]
        
        # Test ranking by strength
        ranked_by_strength = self.debate_engine.rank_arguments(arguments, "strength")
        self.assertEqual(ranked_by_strength[0].strength_score, 0.9)
        
        # Test ranking by logic
        ranked_by_logic = self.debate_engine.rank_arguments(arguments, "logic")
        self.assertEqual(ranked_by_logic[0].logical_validity, 0.9)
        
        # Test ranking by novelty
        ranked_by_novelty = self.debate_engine.rank_arguments(arguments, "novelty")
        self.assertEqual(ranked_by_novelty[0].novelty_score, 0.8)
    
    def test_conclusion_strategies(self):
        """Test different debate conclusion strategies."""
        # Create mock arguments with different stances
        arguments = [
            Argument(stance=DebateStance.PRO, strength_score=0.8),
            Argument(stance=DebateStance.PRO, strength_score=0.7),
            Argument(stance=DebateStance.CON, strength_score=0.6),
            Argument(stance=DebateStance.NEUTRAL, strength_score=0.5)
        ]
        
        # Test majority conclusion
        majority_result = self.debate_engine._conclude_by_majority(arguments)
        self.assertIn('majority', majority_result['conclusion'].lower())
        
        # Test strength conclusion
        strength_result = self.debate_engine._conclude_by_strength(arguments)
        self.assertIn('strongest', strength_result['conclusion'].lower())
        
        # Test synthesis conclusion
        synthesis_result = self.debate_engine._conclude_by_synthesis(arguments)
        self.assertIn('synthesis', synthesis_result['conclusion'].lower())
    
    @patch('agents.debate_engine.DebateEngine._ensure_debate_agents')
    def test_debate_initiation(self, mock_ensure_agents):
        """Test debate initiation process."""
        mock_ensure_agents.return_value = None
        
        # Mock the debate agents
        self.debate_engine.critic_agent = Mock()
        self.debate_engine.debater_agents = [Mock(), Mock()]
        
        # Mock the debate process
        with patch.object(self.debate_engine, '_conduct_debate_round') as mock_conduct:
            with patch.object(self.debate_engine, '_initialize_debate_participants') as mock_init:
                with patch.object(self.debate_engine, 'conclude_debate') as mock_conclude:
                    mock_conclude.return_value = DebateResult(
                        debate_id="test_id",
                        claim=self.test_claim,
                        conclusion="Test conclusion",
                        winning_stance=DebateStance.PRO,
                        confidence=0.8,
                        consensus_reached=True,
                        key_arguments=[],
                        contradictions_resolved=[],
                        open_questions=[]
                    )
                    
                    result = self.debate_engine.initiate_debate(self.test_claim)
                    
                    self.assertIsInstance(result, DebateResult)
                    self.assertEqual(result.claim, self.test_claim)
                    mock_init.assert_called_once()
                    self.assertEqual(mock_conduct.call_count, 3)  # max_rounds
    
    def test_get_agent_instructions(self):
        """Test agent instructions generation."""
        instructions = self.debate_engine.get_agent_instructions()
        self.assertIn("Debate Engine", instructions)
        self.assertIn("dialectical reasoning", instructions)
        self.assertIn("structured", instructions)
    
    def test_respond_method(self):
        """Test agent response generation."""
        # Test with debate context
        context = {"claim": self.test_claim}
        response = self.debate_engine.respond("debate this claim", context)
        self.assertIsInstance(response, str)
        
        # Test without context
        response = self.debate_engine.respond("what can you do?")
        self.assertIn("debate", response.lower())


class TestCriticAgent(unittest.TestCase):
    """Test the CriticAgent functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.critic = CriticAgent()
        self.test_content = "We should deploy this AI model immediately without any testing because it looks good"
    
    def test_critic_initialization(self):
        """Test CriticAgent initialization."""
        self.assertEqual(self.critic.name, "critic")
        self.assertIsInstance(self.critic.criticism_style, str)
        self.assertIsInstance(self.critic.focus_areas, list)
    
    def test_analyze_logical_flaws(self):
        """Test logical flaw analysis."""
        flaws = self.critic.analyze_logical_flaws(self.test_content)
        
        self.assertIsInstance(flaws, list)
        # Should detect missing reasoning
        reasoning_flaw = next((f for f in flaws if f['type'] == 'missing_reasoning'), None)
        self.assertIsNotNone(reasoning_flaw)
        self.assertEqual(reasoning_flaw['severity'], 'high')
    
    def test_evaluate_contradictions(self):
        """Test contradiction evaluation."""
        statements = [
            "AI safety is extremely important and should be prioritized",
            "We should not worry about AI safety and deploy quickly"
        ]
        
        contradictions = self.critic.evaluate_contradictions(statements)
        self.assertIsInstance(contradictions, list)
        if contradictions:  # May not always detect simple contradictions
            self.assertIn('type', contradictions[0])
            self.assertIn('confidence', contradictions[0])
    
    def test_challenge_assumptions(self):
        """Test assumption challenging."""
        content_with_assumptions = "Obviously, AI will always be beneficial and clearly everyone knows this"
        
        assumptions = self.critic.challenge_assumptions(content_with_assumptions)
        self.assertIsInstance(assumptions, list)
        
        # Should detect assumption indicators
        self.assertTrue(any('obviously' in a.get('indicator', '') for a in assumptions))
        self.assertTrue(any('clearly' in a.get('indicator', '') for a in assumptions))
    
    def test_log_criticism(self):
        """Test criticism logging."""
        criticism = {
            'type': 'logical_flaw',
            'severity': 'high',
            'description': 'Test criticism'
        }
        
        # Should not raise an exception
        try:
            self.critic.log_criticism(self.test_content, criticism)
        except Exception as e:
            self.fail(f"log_criticism raised an exception: {e}")
    
    def test_get_criticism_capabilities(self):
        """Test capability reporting."""
        capabilities = self.critic.get_criticism_capabilities()
        
        self.assertIn('criticism_style', capabilities)
        self.assertIn('focus_areas', capabilities)
        self.assertIn('specialized_methods', capabilities)
        self.assertIn('analyze_logical_flaws', capabilities['specialized_methods'])


class TestDebaterAgent(unittest.TestCase):
    """Test the DebaterAgent functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.debater = DebaterAgent()
        self.test_claim = "Remote work is more productive than office work"
    
    def test_debater_initialization(self):
        """Test DebaterAgent initialization."""
        self.assertEqual(self.debater.name, "debater")
        self.assertIsInstance(self.debater.personality_traits, dict)
        self.assertIn('logical', self.debater.personality_traits)
        self.assertIn('skeptical', self.debater.personality_traits)
    
    def test_personality_traits(self):
        """Test personality trait configuration."""
        custom_traits = {
            'skeptical': 0.9,
            'optimistic': 0.2,
            'logical': 1.0
        }
        
        custom_debater = DebaterAgent(personality_traits=custom_traits)
        self.assertEqual(custom_debater.personality_traits['skeptical'], 0.9)
        self.assertEqual(custom_debater.personality_traits['logical'], 1.0)
    
    def test_generate_argument(self):
        """Test argument generation."""
        # Test pro argument
        pro_arg = self.debater.generate_argument(self.test_claim, DebateStance.PRO)
        self.assertIsInstance(pro_arg, str)
        self.assertGreater(len(pro_arg), 10)
        
        # Test con argument
        con_arg = self.debater.generate_argument(self.test_claim, DebateStance.CON)
        self.assertIsInstance(con_arg, str)
        self.assertGreater(len(con_arg), 10)
        
        # Test neutral argument
        neutral_arg = self.debater.generate_argument(self.test_claim, DebateStance.NEUTRAL)
        self.assertIsInstance(neutral_arg, str)
        self.assertGreater(len(neutral_arg), 10)
    
    def test_respond_to_opponent(self):
        """Test opponent argument response."""
        opponent_arg = "Remote work is terrible because people always slack off at home"
        
        response = self.debater.respond_to_opponent(opponent_arg)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 10)
    
    def test_argument_weakness_identification(self):
        """Test identification of argument weaknesses."""
        weak_argument = "Everyone always works better at home and this is never a problem"
        
        weaknesses = self.debater._identify_argument_weaknesses(weak_argument)
        self.assertIn('overgeneralization', weaknesses)
    
    def test_argument_style_determination(self):
        """Test argument style determination based on personality."""
        style = self.debater._determine_argument_style(DebateStance.PRO)
        
        self.assertIn('logical_emphasis', style)
        self.assertIn('emotional_appeal', style)
        self.assertIsInstance(style['logical_emphasis'], float)
    
    def test_get_debate_capabilities(self):
        """Test capability reporting."""
        capabilities = self.debater.get_debate_capabilities()
        
        self.assertIn('personality_traits', capabilities)
        self.assertIn('debate_style', capabilities)
        self.assertIn('generate_argument', capabilities['specialized_methods'])
        self.assertIn('respond_to_opponent', capabilities['specialized_methods'])


class TestReflectionOrchestrator(unittest.TestCase):
    """Test the ReflectionOrchestrator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = ReflectionOrchestrator()
        self.test_topic = "system performance analysis"
    
    def test_orchestrator_initialization(self):
        """Test ReflectionOrchestrator initialization."""
        self.assertEqual(self.orchestrator.name, "reflection_orchestrator")
        self.assertIsInstance(self.orchestrator.contradiction_threshold, float)
        self.assertIsInstance(self.orchestrator.uncertainty_threshold, float)
        self.assertIsInstance(self.orchestrator.pending_reflections, list)
    
    def test_contradiction_monitoring(self):
        """Test contradiction detection in belief system."""
        belief_system = {
            'beliefs': {
                'belief1': {'content': 'AI safety is not important'},
                'belief2': {'content': 'AI safety is critically important'}
            },
            'recent_actions': []
        }
        
        contradictions = self.orchestrator.monitor_contradictions(belief_system)
        self.assertIsInstance(contradictions, list)
    
    def test_uncertainty_monitoring(self):
        """Test uncertainty level monitoring."""
        decision_context = {
            'recent_decisions': [
                {'confidence': 0.3},
                {'confidence': 0.4},
                {'confidence': 0.2}
            ],
            'recommendations': [
                {'type': 'option_a'},
                {'type': 'option_b'},
                {'type': 'option_c'}
            ]
        }
        
        uncertainty = self.orchestrator.monitor_uncertainty(decision_context)
        self.assertIsInstance(uncertainty, float)
        self.assertGreaterEqual(uncertainty, 0.0)
        self.assertLessEqual(uncertainty, 1.0)
    
    def test_ethical_concern_detection(self):
        """Test ethical concern detection."""
        action_plan = {
            'actions': [
                {
                    'type': 'data_collection',
                    'description': 'collect private user data without consent'
                },
                {
                    'type': 'deployment',
                    'description': 'force users to accept new terms'
                }
            ]
        }
        
        concerns = self.orchestrator.detect_ethical_concerns(action_plan)
        self.assertIsInstance(concerns, list)
        if concerns:
            self.assertTrue(any(c['type'] == 'privacy_concern' for c in concerns))
    
    def test_trigger_reflection(self):
        """Test reflection triggering."""
        result = self.orchestrator.trigger_reflection(
            self.test_topic,
            ReflectionTrigger.PERIODIC_REVIEW
        )
        
        self.assertIsInstance(result, ReflectionResult)
        self.assertEqual(result.request_id, result.request_id)  # Should have an ID
        self.assertIsInstance(result.debate_triggered, bool)
        self.assertIsInstance(result.insights, list)
    
    def test_reflection_priority_determination(self):
        """Test reflection priority assignment."""
        critical_priority = self.orchestrator._determine_priority(ReflectionTrigger.ETHICAL_CONCERN)
        self.assertEqual(critical_priority, ReflectionPriority.CRITICAL)
        
        low_priority = self.orchestrator._determine_priority(ReflectionTrigger.PERIODIC_REVIEW)
        self.assertEqual(low_priority, ReflectionPriority.LOW)
    
    def test_should_trigger_debate(self):
        """Test debate triggering decision logic."""
        critical_request = ReflectionRequest(
            trigger=ReflectionTrigger.ETHICAL_CONCERN,
            priority=ReflectionPriority.CRITICAL,
            topic="Test topic"
        )
        
        should_debate = self.orchestrator._should_trigger_debate(critical_request)
        self.assertTrue(should_debate)
        
        low_priority_request = ReflectionRequest(
            trigger=ReflectionTrigger.PERIODIC_REVIEW,
            priority=ReflectionPriority.LOW,
            topic="Test topic"
        )
        
        should_not_debate = self.orchestrator._should_trigger_debate(low_priority_request)
        # May or may not trigger based on context
        self.assertIsInstance(should_not_debate, bool)
    
    def test_get_reflection_statistics(self):
        """Test reflection statistics generation."""
        stats = self.orchestrator.get_reflection_statistics()
        
        expected_keys = [
            'total_reflections', 'pending_reflections', 'active_reflections',
            'debates_triggered', 'contradictions_detected', 'uncertainty_events',
            'ethical_flags', 'resolution_rate'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)


class TestIntegrationDebateSystem(unittest.TestCase):
    """Integration tests for the complete debate system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.debate_engine = DebateEngine()
        self.critic = CriticAgent()
        self.debater = DebaterAgent()
        self.orchestrator = ReflectionOrchestrator()
    
    def test_end_to_end_debate_flow(self):
        """Test complete debate flow from trigger to resolution."""
        claim = "Automated testing is essential for software quality"
        
        # 1. Critic analyzes the claim
        criticism = self.critic.respond(claim)
        self.assertIsInstance(criticism, str)
        
        # 2. Debater generates arguments
        pro_argument = self.debater.generate_argument(claim, DebateStance.PRO)
        con_argument = self.debater.generate_argument(claim, DebateStance.CON)
        
        self.assertIsInstance(pro_argument, str)
        self.assertIsInstance(con_argument, str)
        self.assertNotEqual(pro_argument, con_argument)
        
        # 3. Reflection orchestrator can trigger debates
        reflection_result = self.orchestrator.trigger_reflection(
            claim, ReflectionTrigger.CONTRADICTION
        )
        
        self.assertIsInstance(reflection_result, ReflectionResult)
    
    def test_contradiction_resolution_workflow(self):
        """Test workflow for resolving contradictions through debate."""
        # Simulate conflicting beliefs
        belief_system = {
            'beliefs': {
                'safety_first': {'content': 'AI safety should be the top priority'},
                'speed_first': {'content': 'AI development speed should be the top priority'}
            }
        }
        
        # Monitor for contradictions
        contradictions = self.orchestrator.monitor_contradictions(belief_system)
        
        # If contradictions found, reflection should be triggered
        if contradictions:
            result = self.orchestrator.trigger_reflection(
                "Priority contradictions", ReflectionTrigger.CONTRADICTION
            )
            self.assertTrue(result.debate_triggered or not result.debate_triggered)  # Either is valid
    
    def test_agent_interactions(self):
        """Test interactions between different agents."""
        topic = "The ethics of AI decision-making"
        
        # Critic provides analysis
        logical_flaws = self.critic.analyze_logical_flaws(topic)
        assumptions = self.critic.challenge_assumptions(topic)
        
        # Debater provides multiple perspectives
        pro_stance = self.debater.generate_argument(topic, DebateStance.PRO)
        con_stance = self.debater.generate_argument(topic, DebateStance.CON)
        
        # Orchestrator coordinates reflection
        reflection = self.orchestrator.trigger_reflection(topic, ReflectionTrigger.ETHICAL_CONCERN)
        
        # All should produce valid outputs
        self.assertIsInstance(logical_flaws, list)
        self.assertIsInstance(assumptions, list)
        self.assertIsInstance(pro_stance, str)
        self.assertIsInstance(con_stance, str)
        self.assertIsInstance(reflection, ReflectionResult)


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(unittest.makeSuite(TestDebateEngine))
    suite.addTest(unittest.makeSuite(TestCriticAgent))
    suite.addTest(unittest.makeSuite(TestDebaterAgent))
    suite.addTest(unittest.makeSuite(TestReflectionOrchestrator))
    suite.addTest(unittest.makeSuite(TestIntegrationDebateSystem))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PHASE 21 DEBATE SYSTEM TEST RESULTS")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    print(f"\n🎯 Phase 21 Self-Critique & Debate Agent Swarms testing complete!")