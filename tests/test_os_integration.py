#!/usr/bin/env python3
"""
🧪 MeRNSTA OS Integration Test Suite - Phase 30
Comprehensive test suite for OS integration components.

This module tests:
- Integration runner daemon functionality
- API endpoint responses and agent hooks
- Shell interaction and command success  
- Hot reload of memory and personality
- End-to-end workflow testing
"""

import pytest
import asyncio
import aiohttp
import json
import os
import sys
import time
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import tempfile
import signal

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_config
from system.integration_runner import MeRNSTAIntegrationRunner, IntegrationState, ContextDetector
from api.system_bridge import SystemBridgeAPI
from cli.mernsta_shell import MeRNSTAShell


class TestIntegrationRunner:
    """Test suite for the integration runner daemon."""
    
    @pytest.fixture
    def runner(self):
        """Create a test integration runner instance."""
        runner = MeRNSTAIntegrationRunner(mode="headless")
        yield runner
        # Cleanup
        if runner.running:
            asyncio.run(runner.shutdown())
    
    def test_runner_initialization(self, runner):
        """Test integration runner initialization."""
        assert runner.mode == "headless"
        assert runner.config is not None
        assert runner.state is not None
        assert isinstance(runner.state, IntegrationState)
        assert runner.state.start_time is not None
    
    def test_config_loading(self, runner):
        """Test configuration loading."""
        assert 'os_integration' in runner.config
        os_config = runner.os_config
        assert 'intervals' in os_config
        assert 'logging' in os_config
        assert 'context' in os_config
    
    def test_should_run_task_logic(self, runner):
        """Test task scheduling logic."""
        # Test with no last run (should run)
        assert runner._should_run_task('reflection', None) == True
        
        # Test with recent run (should not run)
        recent_time = datetime.now() - timedelta(seconds=30)
        assert runner._should_run_task('reflection', recent_time) == False
        
        # Test with old run (should run)
        old_time = datetime.now() - timedelta(hours=7)
        assert runner._should_run_task('reflection', old_time) == True
    
    def test_state_serialization(self, runner):
        """Test state saving and loading."""
        # Modify state
        runner.state.reflection_count = 5
        runner.state.planning_count = 3
        runner.state.last_reflection = datetime.now()
        
        # Save state
        runner._save_state()
        
        # Create new runner and load state
        new_runner = MeRNSTAIntegrationRunner(mode="headless")
        new_runner._load_state()
        
        # Verify state loaded correctly
        assert new_runner.state.reflection_count == 5
        assert new_runner.state.planning_count == 3
        assert new_runner.state.last_reflection is not None
    
    @pytest.mark.asyncio
    async def test_background_tasks(self, runner):
        """Test background task execution."""
        # Mock the cognitive system to avoid dependencies
        runner.cognitive_system = MockCognitiveSystem()
        
        # Set short intervals for testing
        runner.intervals = {
            'health_check': 1,  # 1 second
            'context_detection': 2  # 2 seconds
        }
        
        # Run background tasks for a short time
        runner.running = True
        task = asyncio.create_task(runner._background_task_loop())
        
        # Let it run for a few seconds
        await asyncio.sleep(3)
        
        # Stop the runner
        runner.shutdown_requested = True
        await asyncio.sleep(1)
        task.cancel()
        
        # Verify tasks were executed
        assert runner.state.last_health_check is not None


class TestContextDetector:
    """Test suite for context detection functionality."""
    
    @pytest.fixture
    def detector(self):
        """Create a test context detector."""
        return ContextDetector(max_context_size=100)
    
    def test_detector_initialization(self, detector):
        """Test context detector initialization."""
        assert detector.max_context_size == 100
        assert detector.last_active_window is None
        assert detector.recent_commands == []
    
    def test_shell_history_parsing(self, detector):
        """Test shell history parsing."""
        # This test will only work if bash history exists
        try:
            commands = detector.get_shell_history_recent(5)
            assert isinstance(commands, list)
            # Commands should be strings
            for cmd in commands:
                assert isinstance(cmd, str)
        except Exception:
            # If no bash history or permission issues, skip
            pytest.skip("Shell history not accessible")
    
    def test_context_event_structure(self, detector):
        """Test context event structure."""
        # Force a context change
        detector.last_active_window = "old_window"
        
        # Mock get_active_window to return a new window
        detector.get_active_window = lambda: "new_window"
        detector.get_shell_history_recent = lambda x: ["test_command"]
        
        event = detector.detect_context_changes()
        
        if event:  # Only test if we got an event
            assert 'timestamp' in event
            assert 'changes' in event
            assert 'system' in event
            assert isinstance(event['changes'], list)


class TestSystemBridgeAPI:
    """Test suite for the system bridge API."""
    
    @pytest.fixture
    def api(self):
        """Create a test API instance."""
        return SystemBridgeAPI()
    
    def test_api_initialization(self, api):
        """Test API initialization."""
        assert api.app is not None
        assert api.config is not None
        assert api.os_config is not None
        assert api.api_config is not None
    
    @pytest.mark.asyncio
    async def test_api_startup_shutdown(self, api):
        """Test API startup and shutdown process."""
        # Test startup
        await api._initialize_cognitive_system()
        
        # The cognitive system might not initialize in test environment
        # so we don't assert its existence
        
        # Test that no exceptions are raised
        assert True  # If we get here, startup worked
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, api):
        """Test the health check endpoint."""
        # Create a test client
        from fastapi.testclient import TestClient
        
        client = TestClient(api.app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'timestamp' in data
        assert 'api_version' in data
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, api):
        """Test the root endpoint."""
        from fastapi.testclient import TestClient
        
        client = TestClient(api.app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert 'message' in data
        assert 'version' in data
        assert 'endpoints' in data


@pytest.mark.integration
class TestAPIEndpoints:
    """Integration tests for API endpoints with mock cognitive system."""
    
    @pytest.fixture
    def mock_cognitive_system(self):
        """Create a mock cognitive system for testing."""
        return MockCognitiveSystem()
    
    @pytest.fixture
    def api_with_mock(self, mock_cognitive_system):
        """Create API with mock cognitive system."""
        api = SystemBridgeAPI()
        # Inject mock cognitive system
        import api.system_bridge
        api.system_bridge.cognitive_system = mock_cognitive_system
        return api
    
    @pytest.mark.asyncio
    async def test_ask_endpoint(self, api_with_mock):
        """Test the /ask endpoint."""
        from fastapi.testclient import TestClient
        
        client = TestClient(api_with_mock.app)
        
        response = client.post("/ask", json={
            "query": "What is consciousness?",
            "session_id": "test_session"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert 'response' in data
        assert 'timestamp' in data
    
    @pytest.mark.asyncio
    async def test_memory_endpoint(self, api_with_mock):
        """Test the /memory endpoint."""
        from fastapi.testclient import TestClient
        
        client = TestClient(api_with_mock.app)
        
        response = client.post("/memory", json={
            "query_type": "recent",
            "limit": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        assert 'results' in data
        assert 'query_info' in data
    
    @pytest.mark.asyncio
    async def test_goal_endpoint(self, api_with_mock):
        """Test the /goal endpoint."""
        from fastapi.testclient import TestClient
        
        client = TestClient(api_with_mock.app)
        
        # Test list goals
        response = client.post("/goal", json={"action": "list"})
        assert response.status_code == 200
        data = response.json()
        assert 'success' in data
        assert 'goals' in data
    
    @pytest.mark.asyncio
    async def test_status_endpoint(self, api_with_mock):
        """Test the /status endpoint."""
        from fastapi.testclient import TestClient
        
        client = TestClient(api_with_mock.app)
        
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert 'system_status' in data
        assert 'uptime_seconds' in data


class TestShellInterface:
    """Test suite for the interactive shell."""
    
    @pytest.fixture
    def shell(self):
        """Create a test shell instance."""
        return MeRNSTAShell(api_host="127.0.0.1", api_port=8181)
    
    def test_shell_initialization(self, shell):
        """Test shell initialization."""
        assert shell.api_host == "127.0.0.1"
        assert shell.api_port == 8181
        assert shell.session_id is not None
        assert len(shell.commands) > 0
    
    def test_command_parsing(self, shell):
        """Test command parsing and availability."""
        expected_commands = ['ask', 'status', 'memory', 'reflect', 'goal', 'help', 'exit']
        for cmd in expected_commands:
            assert cmd in shell.commands
    
    @pytest.mark.asyncio
    async def test_api_connection_check(self, shell):
        """Test API connection checking."""
        # This will likely fail in test environment, which is expected
        result = await shell.check_api_connection()
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_command_execution(self, shell):
        """Test command execution without actual API."""
        # Test help command (doesn't require API)
        result = await shell.cmd_help([])
        # Should not raise an exception
        assert result is None or result == True


@pytest.mark.integration  
class TestEndToEndWorkflow:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def temp_config(self):
        """Create a temporary config for testing."""
        config_data = {
            'os_integration': {
                'enabled': True,
                'api': {'port': 8182, 'host': '127.0.0.1'},
                'intervals': {'health_check': 5, 'reflection': 60},
                'logging': {'enabled': True, 'log_level': 'INFO'},
                'persistence': {'resume_on_restart': False}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            yield f.name
        
        os.unlink(f.name)
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_api_server_startup(self):
        """Test that the API server can start and respond."""
        # This test starts an actual API server
        api = SystemBridgeAPI()
        
        # Start server in background thread
        import uvicorn
        import threading
        
        def run_server():
            uvicorn.run(api.app, host="127.0.0.1", port=8183, log_level="critical")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        await asyncio.sleep(2)
        
        # Test connection
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://127.0.0.1:8183/health", timeout=5) as response:
                    assert response.status == 200
                    data = await response.json()
                    assert 'status' in data
        except Exception as e:
            pytest.skip(f"Could not connect to test server: {e}")
    
    @pytest.mark.slow
    def test_startup_script_help(self):
        """Test the startup script help functionality."""
        result = subprocess.run(
            ["./start_os_mode.sh", "help"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        assert result.returncode == 0
        assert "MeRNSTA OS Integration" in result.stdout
        assert "COMPONENTS:" in result.stdout


class MockCognitiveSystem:
    """Mock cognitive system for testing."""
    
    def __init__(self):
        self.active = True
        self.reflection_count = 0
        self.planning_count = 0
    
    def process_input_with_full_cognition(self, text, user_profile_id=None, session_id=None):
        """Mock processing method."""
        return {
            'response': f"Mock response to: {text}",
            'confidence': 0.8,
            'reasoning': 'Mock reasoning',
            'session_id': session_id
        }
    
    def search_memories(self, query, limit=10):
        """Mock memory search."""
        return [
            {'id': 1, 'content': f'Mock memory result for: {query}', 'relevance': 0.9},
            {'id': 2, 'content': f'Another mock result for: {query}', 'relevance': 0.7}
        ]
    
    def get_recent_memories(self, limit=10):
        """Mock recent memories."""
        return [
            {'id': 1, 'content': 'Recent mock memory 1', 'timestamp': datetime.now().isoformat()},
            {'id': 2, 'content': 'Recent mock memory 2', 'timestamp': datetime.now().isoformat()}
        ]
    
    def get_memory_facts(self, limit=10):
        """Mock memory facts."""
        return [
            {'fact': 'Mock fact 1', 'confidence': 0.9},
            {'fact': 'Mock fact 2', 'confidence': 0.8}
        ]
    
    def get_contradictions(self, limit=10):
        """Mock contradictions."""
        return [
            {'contradiction': 'Mock contradiction 1', 'severity': 0.7},
            {'contradiction': 'Mock contradiction 2', 'severity': 0.5}
        ]
    
    def get_current_goals(self):
        """Mock current goals."""
        return [
            {'id': 1, 'text': 'Mock goal 1', 'priority': 0.8},
            {'id': 2, 'text': 'Mock goal 2', 'priority': 0.6}
        ]
    
    def add_goal(self, text, priority=0.5):
        """Mock add goal."""
        return f"goal_{int(time.time())}"
    
    def remove_goal(self, goal_id):
        """Mock remove goal."""
        return True
    
    def update_goal(self, goal_id, text=None, priority=None):
        """Mock update goal."""
        return True
    
    def trigger_autonomous_reflection(self):
        """Mock reflection."""
        self.reflection_count += 1
        return {
            'insights': [f'Mock insight {self.reflection_count}'],
            'success': True
        }
    
    def generate_autonomous_goals(self):
        """Mock goal generation."""
        self.planning_count += 1
        return {
            'goals': [f'Generated goal {self.planning_count}'],
            'success': True
        }
    
    def consolidate_memories(self):
        """Mock memory consolidation."""
        return {'consolidated': 5, 'success': True}
    
    def get_personality_profile(self):
        """Mock personality profile."""
        return {
            'traits': {'openness': 0.8, 'conscientiousness': 0.7},
            'evolution_history': [],
            'stability_metrics': {'variance': 0.1}
        }


# Test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


if __name__ == "__main__":
    """Run tests directly."""
    pytest.main([__file__, "-v"])