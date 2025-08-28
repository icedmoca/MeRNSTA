#!/usr/bin/env python3
"""
Test Suite for MeRNSTA Debate Mode

Tests the debate functionality where all agents respond to provide multiple perspectives.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.registry import AgentRegistry

class TestDebateMode:
    """Test the debate mode functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.registry = AgentRegistry()
        
    def test_debate_mode_basic_functionality(self):
        """Test basic debate mode execution."""
        results = self.registry.execute_debate_mode("Should AI replace human jobs?")
        
        # Should get responses from all agents
        assert len(results) >= 4
        
        # Each result should have proper structure
        for result in results:
            assert 'agent' in result
            assert 'response' in result
            assert result['agent'] in ['planner', 'critic', 'debater', 'reflector']
            assert len(result['response']) > 0
            
    def test_debate_responses_are_different(self):
        """Test that different agents provide different perspectives."""
        results = self.registry.execute_debate_mode("What is the best programming language?")
        
        responses = [result['response'] for result in results]
        
        # Responses should be different (not identical)
        unique_responses = set(responses)
        assert len(unique_responses) > 1  # At least some variation
        
        # Each agent should provide their characteristic perspective
        agent_responses = {result['agent']: result['response'] for result in results}
        
        if 'planner' in agent_responses:
            planner_response = agent_responses['planner'].lower()
            # Planner should mention planning/steps/structure
            assert any(word in planner_response for word in ['step', 'plan', 'structure', 'organize'])
            
        if 'critic' in agent_responses:
            critic_response = agent_responses['critic'].lower()
            # Critic should mention issues/problems/analysis
            assert any(word in critic_response for word in ['issue', 'problem', 'flaw', 'analysis', 'risk'])
            
        if 'debater' in agent_responses:
            debater_response = agent_responses['debater'].lower()
            # Debater should mention pros/cons/perspectives
            assert any(word in debater_response for word in ['pros', 'cons', 'perspective', 'argument', 'side'])
            
        if 'reflector' in agent_responses:
            reflector_response = agent_responses['reflector'].lower()
            # Reflector should mention patterns/insights/reflection
            assert any(word in reflector_response for word in ['pattern', 'insight', 'reflect', 'belief', 'deep'])
    
    def test_debate_with_mathematical_query(self):
        """Test debate mode with mathematical queries."""
        results = self.registry.execute_debate_mode("What is 5 * 7?")
        
        assert len(results) >= 4
        
        # All agents should handle mathematical queries appropriately
        for result in results:
            response = result['response'].lower()
            # Should either solve it or discuss the mathematical nature
            assert any(word in response for word in ['35', 'mathematical', 'calculate', 'expression', 'compute'])
    
    def test_debate_with_philosophical_question(self):
        """Test debate mode with philosophical questions."""
        results = self.registry.execute_debate_mode("What is the meaning of life?")
        
        assert len(results) >= 4
        
        # Each agent should provide their unique perspective
        all_responses = " ".join([result['response'] for result in results]).lower()
        
        # Should contain varied philosophical perspectives
        philosophical_terms = ['meaning', 'purpose', 'existence', 'philosophy', 'life', 'human']
        assert sum(term in all_responses for term in philosophical_terms) >= 2
    
    def test_debate_with_practical_problem(self):
        """Test debate mode with practical problem-solving."""
        results = self.registry.execute_debate_mode("How can we reduce climate change?")
        
        assert len(results) >= 4
        
        agent_responses = {result['agent']: result['response'].lower() for result in results}
        
        # Planner should provide structured approach
        if 'planner' in agent_responses:
            planner_resp = agent_responses['planner']
            assert any(word in planner_resp for word in ['step', 'plan', 'strategy', 'approach'])
            
        # Critic should identify challenges
        if 'critic' in agent_responses:
            critic_resp = agent_responses['critic']
            assert any(word in critic_resp for word in ['challenge', 'difficult', 'issue', 'problem'])
            
        # Debater should show multiple viewpoints
        if 'debater' in agent_responses:
            debater_resp = agent_responses['debater']
            assert any(word in debater_resp for word in ['perspective', 'view', 'side', 'argument'])
            
        # Reflector should provide deeper insights
        if 'reflector' in agent_responses:
            reflector_resp = agent_responses['reflector']
            assert any(word in reflector_resp for word in ['insight', 'underlying', 'pattern', 'belief'])
    
    def test_debate_mode_disabled(self):
        """Test behavior when debate mode is disabled."""
        # Temporarily disable debate mode
        original_config = self.registry.config.get('debate_mode', True)
        self.registry.config['debate_mode'] = False
        
        try:
            results = self.registry.execute_debate_mode("Test question")
            assert results == []  # Should return empty list when disabled
        finally:
            # Restore original config
            self.registry.config['debate_mode'] = original_config
    
    def test_debate_error_handling(self):
        """Test debate mode error handling."""
        # Test with empty message
        results = self.registry.execute_debate_mode("")
        
        # Should handle gracefully
        assert isinstance(results, list)
        
        # Test with very long message
        long_message = "A" * 10000
        results = self.registry.execute_debate_mode(long_message)
        
        # Should handle gracefully
        assert isinstance(results, list)
    
    def test_debate_consistency(self):
        """Test that debate results are reasonably consistent."""
        question = "Should we use renewable energy?"
        
        # Run debate multiple times
        results1 = self.registry.execute_debate_mode(question)
        results2 = self.registry.execute_debate_mode(question)
        
        # Should have same number of agents responding
        assert len(results1) == len(results2)
        
        # Should have same agent names (order might differ)
        agents1 = {result['agent'] for result in results1}
        agents2 = {result['agent'] for result in results2}
        assert agents1 == agents2
        
    def test_debate_with_context(self):
        """Test debate mode with additional context."""
        context = {"urgency": "high", "domain": "technology"}
        results = self.registry.execute_debate_mode("Should we adopt AI in healthcare?", context)
        
        assert len(results) >= 4
        
        # Context should influence responses (at least some agents should consider it)
        all_responses = " ".join([result['response'] for result in results]).lower()
        # Should contain healthcare-related terms
        assert any(word in all_responses for word in ['health', 'medical', 'patient', 'care'])

class TestDebateQuality:
    """Test the quality and diversity of debate responses."""
    
    def setup_method(self):
        """Set up test environment."""
        self.registry = AgentRegistry()
    
    def test_response_length_variation(self):
        """Test that responses have reasonable length variation."""
        results = self.registry.execute_debate_mode("Explain quantum computing")
        
        response_lengths = [len(result['response']) for result in results]
        
        # All responses should be substantial
        assert all(length > 50 for length in response_lengths)
        
        # Should have some variation in length
        min_length = min(response_lengths)
        max_length = max(response_lengths)
        assert max_length > min_length * 1.2  # At least 20% variation
    
    def test_agent_specialization_evident(self):
        """Test that each agent's specialization is evident in responses."""
        results = self.registry.execute_debate_mode("Design a new social media platform")
        
        agent_keywords = {
            'planner': ['step', 'plan', 'phase', 'timeline', 'strategy', 'approach'],
            'critic': ['risk', 'problem', 'issue', 'flaw', 'concern', 'challenge'],
            'debater': ['perspective', 'pro', 'con', 'argument', 'viewpoint', 'side'],
            'reflector': ['insight', 'pattern', 'underlying', 'belief', 'meaning', 'reflect']
        }
        
        for result in results:
            agent_name = result['agent']
            response = result['response'].lower()
            
            if agent_name in agent_keywords:
                keywords = agent_keywords[agent_name]
                # Should contain at least some characteristic keywords
                matches = sum(1 for keyword in keywords if keyword in response)
                assert matches >= 1, f"{agent_name} response doesn't show specialization"
    
    def test_complementary_perspectives(self):
        """Test that agents provide complementary rather than redundant perspectives."""
        results = self.registry.execute_debate_mode("Should we colonize Mars?")
        
        # Get all unique concepts mentioned
        all_concepts = set()
        
        for result in results:
            response = result['response'].lower()
            # Extract key concepts (simplified)
            words = response.split()
            concepts = [word for word in words if len(word) > 4 and word.isalpha()]
            all_concepts.update(concepts)
        
        # Should have a diverse range of concepts
        assert len(all_concepts) > 20  # Rich vocabulary indicates diverse perspectives
        
    def test_debate_addresses_question(self):
        """Test that all agents actually address the original question."""
        question = "What are the ethical implications of AI surveillance?"
        results = self.registry.execute_debate_mode(question)
        
        key_terms = ['ethical', 'ethics', 'surveillance', 'ai', 'artificial', 'moral', 'privacy']
        
        for result in results:
            response = result['response'].lower()
            # Each response should mention at least some key terms from the question
            matches = sum(1 for term in key_terms if term in response)
            assert matches >= 1, f"{result['agent']} response doesn't address the question"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 