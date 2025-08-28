#!/usr/bin/env python3
"""
Integration test for MeRNSTA Unified Startup (python main.py run)

This test verifies that the unified "run" command properly starts all components:
- Web Chat UI server
- REST API server  
- Background cognitive tasks
- Multi-agent system
- Health monitoring

The test performs a full system startup, verifies endpoints are responsive,
and confirms background tasks are running.
"""

import os
import sys
import time
import asyncio
import pytest
import requests
import subprocess
import signal
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from system.unified_runner import MeRNSTAUnifiedRunner


class TestUnifiedStartup:
    """Test suite for unified system startup."""
    
    @pytest.fixture
    def unused_ports(self):
        """Get two unused ports for testing."""
        import socket
        
        def get_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                s.listen(1)
                port = s.getsockname()[1]
            return port
        
        return get_free_port(), get_free_port()
    
    @pytest.mark.asyncio
    async def test_unified_runner_initialization(self):
        """Test that the unified runner can be initialized with various configurations."""
        
        # Test default configuration
        runner = MeRNSTAUnifiedRunner()
        assert runner.web_port == 8000
        assert runner.api_port == 8001
        assert runner.enable_web is True
        assert runner.enable_api is True
        assert runner.enable_background is True
        assert runner.enable_agents is True
        assert runner.enable_enterprise is False
        
        # Test custom configuration
        runner = MeRNSTAUnifiedRunner(
            web_port=9000,
            api_port=9001,
            enable_web=False,
            enable_api=False,
            enable_background=False,
            enable_agents=False,
            enable_enterprise=True,
            debug=True
        )
        assert runner.web_port == 9000
        assert runner.api_port == 9001
        assert runner.enable_web is False
        assert runner.enable_api is False
        assert runner.enable_background is False
        assert runner.enable_agents is False
        assert runner.enable_enterprise is True
        assert runner.debug is True
    
    @pytest.mark.asyncio
    async def test_cognitive_system_initialization(self, unused_ports):
        """Test that the cognitive system initializes properly."""
        web_port, api_port = unused_ports
        
        runner = MeRNSTAUnifiedRunner(
            web_port=web_port,
            api_port=api_port,
            enable_web=False,  # Disable servers for faster test
            enable_api=False,
            enable_background=False,
            enable_enterprise=False
        )
        
        # Test cognitive system initialization
        success = await runner._initialize_cognitive_system()
        assert success is True
        assert runner.cognitive_system is not None
        assert runner.state.cognitive_system_started is True
        
        # Test agent initialization
        success = await runner._initialize_agents()
        assert success is True
        assert runner.state.agents_initialized is True
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_server_startup(self, unused_ports):
        """Test that web and API servers start properly."""
        web_port, api_port = unused_ports
        
        runner = MeRNSTAUnifiedRunner(
            web_port=web_port,
            api_port=api_port,
            enable_background=False,  # Disable background tasks for faster test
            enable_enterprise=False,
            debug=True
        )
        
        # Initialize core system first
        await runner._initialize_cognitive_system()
        await runner._initialize_agents()
        
        # Start web server
        success = await runner._start_web_server()
        assert success is True
        assert runner.state.web_server_started is True
        
        # Start API server  
        success = await runner._start_api_server()
        assert success is True
        assert runner.state.api_server_started is True
        
        # Wait for servers to be ready
        await asyncio.sleep(3)
        
        # Test web server endpoint
        try:
            response = requests.get(f"http://localhost:{web_port}/health", timeout=5)
            assert response.status_code == 200
        except Exception as e:
            pytest.fail(f"Web server health check failed: {e}")
        
        # Test API server endpoint
        try:
            response = requests.get(f"http://127.0.0.1:{api_port}/health", timeout=5)
            assert response.status_code == 200
        except Exception as e:
            pytest.fail(f"API server health check failed: {e}")
        
        # Cleanup
        await runner.shutdown()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_background_tasks_startup(self, unused_ports):
        """Test that background tasks start properly."""
        web_port, api_port = unused_ports
        
        runner = MeRNSTAUnifiedRunner(
            web_port=web_port,
            api_port=api_port,
            enable_web=False,  # Disable servers for faster test
            enable_api=False,
            enable_enterprise=False
        )
        
        # Initialize core system first
        await runner._initialize_cognitive_system()
        await runner._initialize_agents()
        
        # Start background tasks
        success = await runner._start_background_tasks()
        assert success is True
        assert runner.state.background_tasks_started is True
        assert len(runner.state.active_background_tasks) > 0
        
        # Verify expected background tasks are registered
        expected_tasks = {
            'reflection', 'planning', 'memory_consolidation', 
            'health_check', 'context_detection'
        }
        assert expected_tasks.issubset(runner.state.active_background_tasks)
        
        # Cleanup
        await runner.shutdown()
    
    @pytest.mark.asyncio
    def test_status_reporting(self):
        """Test that status reporting works correctly."""
        runner = MeRNSTAUnifiedRunner(
            web_port=8888,
            api_port=8889,
            enable_enterprise=True,
            debug=True
        )
        
        status = runner.get_status()
        
        # Verify status structure
        assert 'running' in status
        assert 'uptime_seconds' in status
        assert 'components' in status
        assert 'configuration' in status
        assert 'active_background_tasks' in status
        assert 'state' in status
        
        # Verify component status
        components = status['components']
        assert 'cognitive_system' in components
        assert 'web_server' in components
        assert 'api_server' in components
        assert 'background_tasks' in components
        assert 'agents' in components
        assert 'enterprise' in components
        
        # Verify configuration
        config = status['configuration']
        assert config['web_port'] == 8888
        assert config['api_port'] == 8889
        assert config['enable_enterprise'] is True
        assert config['debug'] is True
    
    def test_main_py_run_command_parsing(self):
        """Test that main.py properly parses the 'run' command arguments."""
        # This test verifies argument parsing without actually starting the system
        
        import main
        
        # Test that 'run' is in the valid choices
        parser = main.create_parser()
        
        # Test basic run command
        args = parser.parse_args(['run'])
        assert args.mode == 'run'
        
        # Test run command with options
        args = parser.parse_args([
            'run', 
            '--web-port', '9000',
            '--api-port', '9001',
            '--no-web',
            '--no-api',
            '--no-background',
            '--no-agents',
            '--enterprise',
            '--debug'
        ])
        
        assert args.mode == 'run'
        assert args.web_port == 9000
        assert args.api_port == 9001
        assert args.no_web is True
        assert args.no_api is True
        assert args.no_background is True
        assert args.no_agents is True
        assert args.enterprise is True
        assert args.debug is True


class TestUnifiedStartupIntegration:
    """Integration tests that run the full system startup."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_system_startup_via_subprocess(self):
        """Test complete system startup using subprocess call to main.py run."""
        
        # Find unused ports
        import socket
        def get_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                s.listen(1)
                port = s.getsockname()[1]
            return port
        
        web_port = get_free_port()
        api_port = get_free_port()
        
        # Start the system in a subprocess
        cmd = [
            sys.executable, 'main.py', 'run',
            '--web-port', str(web_port),
            '--api-port', str(api_port),
            '--no-enterprise',  # Disable enterprise features for test
            '--debug'
        ]
        
        process = None
        try:
            # Start the process
            process = subprocess.Popen(
                cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for startup (allow up to 60 seconds)
            startup_timeout = 60
            web_ready = False
            api_ready = False
            
            for _ in range(startup_timeout):
                time.sleep(1)
                
                # Check if process is still running
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    pytest.fail(f"Process exited early. STDOUT: {stdout}, STDERR: {stderr}")
                
                # Test web server
                if not web_ready:
                    try:
                        response = requests.get(f"http://localhost:{web_port}/health", timeout=2)
                        if response.status_code == 200:
                            web_ready = True
                            print(f"✅ Web server ready on port {web_port}")
                    except requests.RequestException:
                        pass
                
                # Test API server
                if not api_ready:
                    try:
                        response = requests.get(f"http://127.0.0.1:{api_port}/health", timeout=2)
                        if response.status_code == 200:
                            api_ready = True
                            print(f"✅ API server ready on port {api_port}")
                    except requests.RequestException:
                        pass
                
                # If both are ready, test functionality
                if web_ready and api_ready:
                    print("🚀 Both servers are ready - testing functionality...")
                    
                    # Test web chat interface
                    try:
                        response = requests.get(f"http://localhost:{web_port}/chat", timeout=5)
                        assert response.status_code == 200
                        print("✅ Web chat interface accessible")
                    except Exception as e:
                        pytest.fail(f"Web chat interface test failed: {e}")
                    
                    # Test API functionality
                    try:
                        response = requests.post(
                            f"http://127.0.0.1:{api_port}/ask",
                            json={"query": "Hello, test query"},
                            timeout=10
                        )
                        # Note: This might fail if cognitive system isn't fully initialized,
                        # but we should at least get a proper HTTP response
                        assert response.status_code in [200, 503]  # 503 if cognitive system not ready
                        print("✅ API endpoint responds")
                    except Exception as e:
                        pytest.fail(f"API endpoint test failed: {e}")
                    
                    # Test successful!
                    print("🎉 Full system startup test PASSED!")
                    return
            
            # If we get here, startup timed out
            pytest.fail(f"System startup timed out after {startup_timeout} seconds. Web ready: {web_ready}, API ready: {api_ready}")
            
        finally:
            # Cleanup: terminate the process
            if process and process.poll() is None:
                print("🛑 Terminating test process...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                    print("✅ Process terminated gracefully")
                except subprocess.TimeoutExpired:
                    print("⚠️ Process did not terminate gracefully, killing...")
                    process.kill()
                    process.wait()
    
    @pytest.mark.integration
    @pytest.mark.slow 
    def test_unified_startup_with_graceful_shutdown(self):
        """Test that the unified system starts up and shuts down gracefully."""
        
        # Find unused ports
        import socket
        def get_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                s.listen(1)
                port = s.getsockname()[1]
            return port
        
        web_port = get_free_port()
        api_port = get_free_port()
        
        async def run_test():
            runner = MeRNSTAUnifiedRunner(
                web_port=web_port,
                api_port=api_port,
                enable_enterprise=False,  # Disable for faster startup
                debug=True
            )
            
            # Start the system
            startup_task = asyncio.create_task(runner.start())
            
            # Wait for startup
            await asyncio.sleep(10)
            
            # Verify system is running
            assert runner.running is True
            
            # Test endpoints
            response = requests.get(f"http://localhost:{web_port}/health", timeout=5)
            assert response.status_code == 200
            
            response = requests.get(f"http://127.0.0.1:{api_port}/health", timeout=5)
            assert response.status_code == 200
            
            # Request shutdown
            await runner.shutdown()
            
            # Verify shutdown
            assert runner.state.graceful_shutdown_requested is True
            assert runner.running is False
            
            # Cancel the startup task
            startup_task.cancel()
            try:
                await startup_task
            except asyncio.CancelledError:
                pass
        
        # Run the async test
        asyncio.run(run_test())


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])
