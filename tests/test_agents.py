#!/usr/bin/env python3
"""
Test Suite for MeRNSTA Multi-Agent Cognitive System

Tests each agent's responses, capabilities, and specialized behaviors.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.registry import AgentRegistry, get_agent_registry
from agents.planner import PlannerAgent
from agents.critic import CriticAgent
from agents.debater import DebaterAgent
from agents.reflector import ReflectorAgent

class TestAgentRegistry:
    """Test the agent registry functionality."""
    
    def test_registry_initialization(self):
        """Test that the registry initializes correctly."""
        registry = AgentRegistry()
        assert registry is not None
        assert isinstance(registry.agents, dict)
        
    def test_registry_loads_agents(self):
        """Test that agents are loaded correctly."""
        registry = AgentRegistry()
        
        # Should have all 4 core agents
        expected_agents = ['planner', 'critic', 'debater', 'reflector']
        for agent_name in expected_agents:
            assert agent_name in registry.agents
            assert registry.is_agent_available(agent_name)
            
    def test_get_agent(self):
        """Test retrieving specific agents."""
        registry = AgentRegistry()
        
        planner = registry.get_agent('planner')
        assert planner is not None
        assert isinstance(planner, PlannerAgent)
        
        critic = registry.get_agent('critic')
        assert critic is not None
        assert isinstance(critic, CriticAgent)
        
        # Test non-existent agent
        fake_agent = registry.get_agent('nonexistent')
        assert fake_agent is None
        
    def test_system_status(self):
        """Test system status reporting."""
        registry = AgentRegistry()
        status = registry.get_system_status()
        
        assert 'enabled' in status
        assert 'total_agents' in status
        assert 'agent_names' in status
        assert 'debate_mode' in status
        assert status['total_agents'] >= 4  # At least our 4 core agents

class TestPlannerAgent:
    """Test the PlannerAgent functionality."""
    
    def test_planner_initialization(self):
        """Test planner agent initializes correctly."""
        agent = PlannerAgent()
        assert agent.name == "planner"
        assert hasattr(agent, 'planning_style')
        assert hasattr(agent, 'max_steps')
        
    def test_planner_responds_to_tasks(self):
        """Test planner responds to task planning requests."""
        agent = PlannerAgent()
        
        response = agent.respond("Plan a birthday party")
        assert response is not None
        assert len(response) > 0
        assert isinstance(response, str)
        # Should contain planning elements
        assert any(keyword in response.lower() for keyword in ['step', 'plan', '📋', 'task'])
        
    def test_planner_handles_math_tasks(self):
        """Test planner handles mathematical tasks specifically."""
        agent = PlannerAgent()
        
        response = agent.respond("what is 5+3")
        assert response is not None
        assert "mathematical" in response.lower() or "math" in response.lower()
        
    def test_planner_capabilities(self):
        """Test planner capabilities reporting."""
        agent = PlannerAgent()
        caps = agent.get_capabilities()
        
        assert caps['name'] == 'planner'
        assert 'enabled' in caps
        assert 'has_llm' in caps
        assert 'has_symbolic' in caps

class TestCriticAgent:
    """Test the CriticAgent functionality."""
    
    def test_critic_initialization(self):
        """Test critic agent initializes correctly."""
        agent = CriticAgent()
        assert agent.name == "critic"
        assert hasattr(agent, 'criticism_style')
        assert hasattr(agent, 'focus_areas')
        
    def test_critic_responds_to_content(self):
        """Test critic responds to content analysis requests."""
        agent = CriticAgent()
        
        response = agent.respond("This plan will definitely work perfectly with no issues")
        assert response is not None
        assert len(response) > 0
        # Should contain critical analysis elements
        assert any(keyword in response.lower() for keyword in ['issue', 'flaw', 'problem', 'risk', '🔍'])
        
    def test_critic_analyzes_math(self):
        """Test critic analyzes mathematical expressions."""
        agent = CriticAgent()
        
        response = agent.respond("2+2")
        assert response is not None
        assert "mathematical" in response.lower() or "analysis" in response.lower()
        
    def test_critic_specialized_methods(self):
        """Test critic's specialized analysis methods."""
        agent = CriticAgent()
        
        # Test assumption analysis
        assumptions = agent.analyze_assumptions("Everyone loves chocolate")
        assert assumptions is not None
        assert "assumption" in assumptions.lower()
        
        # Test risk identification
        risks = agent.identify_risks("Launch rocket without testing")
        assert risks is not None
        assert "risk" in risks.lower()

class TestDebaterAgent:
    """Test the DebaterAgent functionality."""
    
    def test_debater_initialization(self):
        """Test debater agent initializes correctly."""
        agent = DebaterAgent()
        assert agent.name == "debater"
        assert hasattr(agent, 'debate_style')
        assert hasattr(agent, 'perspective_count')
        
    def test_debater_explores_perspectives(self):
        """Test debater explores multiple perspectives."""
        agent = DebaterAgent()
        
        response = agent.respond("Should AI be regulated?")
        assert response is not None
        assert len(response) > 0
        # Should contain debate elements
        assert any(keyword in response.lower() for keyword in ['perspective', 'pro', 'con', 'argument', '⚖️'])
        
    def test_debater_handles_math_topics(self):
        """Test debater handles mathematical topics."""
        agent = DebaterAgent()
        
        response = agent.respond("3*4")
        assert response is not None
        assert "mathematical" in response.lower() or "debate" in response.lower()
        
    def test_debater_specialized_methods(self):
        """Test debater's specialized methods."""
        agent = DebaterAgent()
        
        # Test devil's advocate
        devils_advocate = agent.devil_advocate("AI is always beneficial")
        assert devils_advocate is not None
        assert "devil" in devils_advocate.lower() or "advocate" in devils_advocate.lower()
        
        # Test common ground finding
        common_ground = agent.find_common_ground("AI is good", "AI is dangerous")
        assert common_ground is not None
        assert "common" in common_ground.lower() or "ground" in common_ground.lower()

class TestReflectorAgent:
    """Test the ReflectorAgent functionality."""
    
    def test_reflector_initialization(self):
        """Test reflector agent initializes correctly."""
        agent = ReflectorAgent()
        assert agent.name == "reflector"
        assert hasattr(agent, 'reflection_depth')
        assert hasattr(agent, 'focus_patterns')
        
    def test_reflector_provides_insights(self):
        """Test reflector provides reflective insights."""
        agent = ReflectorAgent()
        
        response = agent.respond("I keep making the same mistakes over and over")
        assert response is not None
        assert len(response) > 0
        # Should contain reflective elements
        assert any(keyword in response.lower() for keyword in ['pattern', 'insight', 'reflection', 'belief', '🌟'])
        
    def test_reflector_analyzes_math_thinking(self):
        """Test reflector analyzes mathematical thinking patterns."""
        agent = ReflectorAgent()
        
        response = agent.respond("7-2")
        assert response is not None
        assert "mathematical" in response.lower() or "thinking" in response.lower()
        
    def test_reflector_specialized_methods(self):
        """Test reflector's specialized methods."""
        agent = ReflectorAgent()
        
        # Test belief summarization
        beliefs = agent.summarize_belief_system("technology")
        assert beliefs is not None
        assert "belief" in beliefs.lower()
        
        # Test cognitive pattern identification
        patterns = agent.identify_cognitive_patterns()
        assert patterns is not None
        assert "pattern" in patterns.lower() or "cognitive" in patterns.lower()
        
        # Test insight synthesis
        insights = agent.synthesize_insights("learning")
        assert insights is not None
        assert "insight" in insights.lower()

class TestAgentIntegration:
    """Test integration between agents and the registry."""
    
    def test_debate_mode(self):
        """Test debate mode functionality."""
        registry = AgentRegistry()
        
        results = registry.execute_debate_mode("Should we use AI in education?")
        assert results is not None
        assert len(results) >= 4  # Should have responses from all agents
        
        # Each result should have agent and response
        for result in results:
            assert 'agent' in result
            assert 'response' in result
            assert result['agent'] in ['planner', 'critic', 'debater', 'reflector']
            assert len(result['response']) > 0
            
    def test_agent_capabilities_reporting(self):
        """Test that all agents report their capabilities correctly."""
        registry = AgentRegistry()
        
        all_caps = registry.get_agent_capabilities()
        assert len(all_caps) >= 4
        
        for agent_name in ['planner', 'critic', 'debater', 'reflector']:
            assert agent_name in all_caps
            caps = all_caps[agent_name]
            assert 'name' in caps
            assert 'enabled' in caps
            assert caps['name'] == agent_name
            
    def test_global_registry_access(self):
        """Test global registry access functions."""
        registry = get_agent_registry()
        assert registry is not None
        assert len(registry) >= 4  # Should have at least 4 agents
        
        # Test iteration
        agent_names = list(registry)
        assert 'planner' in agent_names
        assert 'critic' in agent_names
        
        # Test containment
        assert 'planner' in registry
        assert 'nonexistent' not in registry

class TestAgentErrorHandling:
    """Test error handling in agents."""
    
    def test_agent_handles_empty_input(self):
        """Test agents handle empty input gracefully."""
        registry = AgentRegistry()
        
        for agent_name in registry.get_agent_names():
            agent = registry.get_agent(agent_name)
            response = agent.respond("")
            assert response is not None
            # Should handle gracefully, not crash
            
    def test_agent_handles_invalid_context(self):
        """Test agents handle invalid context gracefully."""
        registry = AgentRegistry()
        
        agent = registry.get_agent('planner')
        response = agent.respond("Plan something", context={"invalid": "context"})
        assert response is not None
        # Should handle gracefully, not crash

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 