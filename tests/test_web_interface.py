#!/usr/bin/env python3
"""
Test Suite for MeRNSTA Web Interface

Tests the FastAPI endpoints, chat functionality, and web integration.
"""

import pytest
import sys
import os
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the FastAPI app
from web.main import app

class TestWebApplication:
    """Test the main FastAPI application."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "MeRNSTA" in data["message"]
        assert "chat_url" in data
        
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "agents" in data
        assert "enabled" in data
        
    def test_agents_list_endpoint(self):
        """Test the agents listing endpoint."""
        response = self.client.get("/agents")
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert "capabilities" in data
        assert "system_status" in data
        
        # Should have our core agents
        assert len(data["agents"]) >= 4
        expected_agents = ['planner', 'critic', 'debater', 'reflector']
        for agent in expected_agents:
            assert agent in data["agents"]
            
    def test_chat_interface_endpoint(self):
        """Test the chat interface endpoint."""
        response = self.client.get("/chat")
        assert response.status_code == 200
        # Should return HTML content
        assert "text/html" in response.headers.get("content-type", "")
        assert "MeRNSTA" in response.text

class TestAgentEndpoints:
    """Test the agent-specific endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_agent_respond_single_agent(self):
        """Test single agent response endpoint."""
        request_data = {
            "agent": "planner",
            "message": "Plan a simple task",
            "debate": False
        }
        
        response = self.client.post("/agents/respond", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "agent" in data
        assert "response" in data
        assert data["agent"] == "planner"
        assert len(data["response"]) > 0
        
    def test_agent_respond_debate_mode(self):
        """Test debate mode response endpoint."""
        request_data = {
            "message": "What are the pros and cons of remote work?",
            "debate": True
        }
        
        response = self.client.post("/agents/respond", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "agent" in data
        assert "response" in data
        assert data["agent"] == "debate"
        assert len(data["response"]) > 0
        # Should contain multiple agent responses
        assert "Agent" in data["response"]  # Should have agent names in response
        
    def test_agent_respond_math_query(self):
        """Test mathematical query handling."""
        request_data = {
            "agent": "planner",
            "message": "what is 3+4",
            "debate": False
        }
        
        response = self.client.post("/agents/respond", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data
        # Should handle mathematical queries appropriately
        
    def test_agent_respond_conversational_query(self):
        """Test conversational query handling."""
        request_data = {
            "agent": "reflector",
            "message": "how are you feeling today?",
            "debate": False
        }
        
        response = self.client.post("/agents/respond", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data
        assert len(data["response"]) > 0
        
    def test_agent_respond_invalid_agent(self):
        """Test response with invalid agent name."""
        request_data = {
            "agent": "nonexistent",
            "message": "test message",
            "debate": False
        }
        
        response = self.client.post("/agents/respond", json=request_data)
        assert response.status_code == 404
        
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
        
    def test_agent_respond_empty_message(self):
        """Test response with empty message."""
        request_data = {
            "agent": "planner",
            "message": "",
            "debate": False
        }
        
        response = self.client.post("/agents/respond", json=request_data)
        # Should handle gracefully (may return 200 with appropriate response)
        assert response.status_code in [200, 400]
        
    def test_agent_status_endpoint(self):
        """Test agent status endpoint."""
        response = self.client.get("/agents/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "enabled" in data
        assert "total_agents" in data
        assert "agent_names" in data
        assert "debate_mode" in data
        
    def test_agent_capabilities_endpoint(self):
        """Test agent capabilities endpoint."""
        response = self.client.get("/agents/capabilities")
        assert response.status_code == 200
        
        data = response.json()
        # Should return capabilities for all agents
        assert len(data) >= 4
        
        # Test specific agent capabilities
        response = self.client.get("/agents/capabilities?agent=planner")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert data["name"] == "planner"

class TestRequestValidation:
    """Test request validation and error handling."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_invalid_json_request(self):
        """Test handling of invalid JSON requests."""
        response = self.client.post("/agents/respond", data="invalid json")
        assert response.status_code == 422  # Unprocessable Entity
        
    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        # Missing message field
        request_data = {
            "agent": "planner",
            "debate": False
        }
        
        response = self.client.post("/agents/respond", json=request_data)
        assert response.status_code == 422
        
    def test_invalid_field_types(self):
        """Test handling of invalid field types."""
        request_data = {
            "agent": 123,  # Should be string
            "message": "test",
            "debate": "not_a_boolean"  # Should be boolean
        }
        
        response = self.client.post("/agents/respond", json=request_data)
        assert response.status_code == 422

class TestAgentSpecificBehaviors:
    """Test agent-specific behaviors through the web interface."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_planner_agent_response(self):
        """Test planner agent specific response format."""
        request_data = {
            "agent": "planner",
            "message": "Plan a weekend trip",
            "debate": False
        }
        
        response = self.client.post("/agents/respond", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        # Planner responses should contain planning elements
        response_text = data["response"].lower()
        assert any(word in response_text for word in ["step", "plan", "task", "📋"])
        
    def test_critic_agent_response(self):
        """Test critic agent specific response format."""
        request_data = {
            "agent": "critic",
            "message": "This is a perfect plan with no flaws",
            "debate": False
        }
        
        response = self.client.post("/agents/respond", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        # Critic responses should contain critical analysis
        response_text = data["response"].lower()
        assert any(word in response_text for word in ["analysis", "issue", "critical", "🔍"])
        
    def test_debater_agent_response(self):
        """Test debater agent specific response format."""
        request_data = {
            "agent": "debater",
            "message": "Should we adopt renewable energy?",
            "debate": False
        }
        
        response = self.client.post("/agents/respond", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        # Debater responses should contain multiple perspectives
        response_text = data["response"].lower()
        assert any(word in response_text for word in ["perspective", "debate", "pros", "cons", "⚖️"])
        
    def test_reflector_agent_response(self):
        """Test reflector agent specific response format."""
        request_data = {
            "agent": "reflector",
            "message": "I keep making the same mistakes",
            "debate": False
        }
        
        response = self.client.post("/agents/respond", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        # Reflector responses should contain reflective analysis
        response_text = data["response"].lower()
        assert any(word in response_text for word in ["reflection", "pattern", "insight", "belief", "🌟"])

class TestDebateMode:
    """Test debate mode functionality specifically."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_debate_mode_all_agents_respond(self):
        """Test that debate mode gets responses from all agents."""
        request_data = {
            "message": "What is the future of artificial intelligence?",
            "debate": True
        }
        
        response = self.client.post("/agents/respond", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["agent"] == "debate"
        
        # Should contain responses from all agents
        response_text = data["response"]
        expected_agents = ["Planner", "Critic", "Debater", "Reflector"]
        for agent in expected_agents:
            assert agent in response_text
            
    def test_debate_mode_with_math_query(self):
        """Test debate mode with mathematical queries."""
        request_data = {
            "message": "What is 2+2?",
            "debate": True
        }
        
        response = self.client.post("/agents/respond", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        # All agents should provide their perspective on the mathematical query

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 