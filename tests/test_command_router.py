#!/usr/bin/env python3
"""
Tests for Phase 13: Command Routing & Execution System

Comprehensive tests for the command router, tool registry, and autonomous command generation.
"""

import pytest
import asyncio
import tempfile
import os
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Import the Phase 13 components
from agents.command_router import CommandRouter, get_command_router, route_command, route_command_async
from agents.registry import ToolRegistry, get_tool_registry, AgentRegistry, get_agent_registry
from storage.tool_use_log import ToolUseLogger, get_tool_logger


class TestCommandRouter:
    """Test the main CommandRouter class."""
    
    @pytest.fixture
    def router(self):
        """Create a CommandRouter instance for testing."""
        return CommandRouter()
    
    @pytest.fixture
    def tool_registry(self):
        """Create a ToolRegistry instance for testing."""
        return ToolRegistry()
    
    @pytest.fixture
    def agent_registry(self):
        """Create an AgentRegistry instance for testing."""
        return AgentRegistry()
    
    def test_command_router_initialization(self, router):
        """Test CommandRouter initialization."""
        assert router is not None
        assert router.unrestricted_mode is True
        assert router.allow_all_shell_commands is True
        assert router.enable_pip_install is True
        assert router.log_every_execution is True
        assert router.command_history == []
    
    def test_command_parsing_shell(self, router):
        """Test parsing of shell commands."""
        command = '/run_shell "ls -la"'
        parsed = router.parse_command(command)
        
        assert parsed['type'] == 'shell'
        assert parsed['args']['command'] == 'ls -la'
        assert parsed['args']['raw'] is True
    
    def test_command_parsing_pip(self, router):
        """Test parsing of pip install commands."""
        command = '/pip_install requests'
        parsed = router.parse_command(command)
        
        assert parsed['type'] == 'pip_install'
        assert parsed['args']['package'] == 'requests'
        assert parsed['args']['upgrade'] is False
    
    def test_command_parsing_tool(self, router):
        """Test parsing of tool commands."""
        command = '/run_tool read_file test.txt'
        parsed = router.parse_command(command)
        
        assert parsed['type'] == 'tool'
        assert parsed['args']['tool_name'] == 'read_file'
        assert parsed['args']['tool_args'] == 'test.txt'
    
    def test_command_parsing_invalid(self, router):
        """Test parsing of invalid commands."""
        command = 'invalid_command'
        parsed = router.parse_command(command)
        
        assert parsed['type'] == 'invalid'
        assert 'error' in parsed
    
    @pytest.mark.asyncio
    async def test_shell_command_execution(self, router):
        """Test shell command execution."""
        command = '/run_shell "echo hello"'
        result = await router.execute_command(command, "test")
        
        assert result['success'] is True
        assert 'hello' in result['output']
        assert result['executor'] == 'test'
        assert result['command'] == command
    
    @pytest.mark.asyncio
    async def test_shell_command_failure(self, router):
        """Test shell command failure handling."""
        command = '/run_shell "nonexistent_command_xyz"'
        result = await router.execute_command(command, "test")
        
        assert result['success'] is False
        assert 'error' in result
        assert result['executor'] == 'test'
    
    def test_command_history(self, router):
        """Test command history tracking."""
        initial_count = len(router.command_history)
        
        # Add a mock command to history
        router.command_history.append({
            'command': '/test_command',
            'executor': 'test',
            'result': {'success': True},
            'timestamp': datetime.now().isoformat()
        })
        
        assert len(router.command_history) == initial_count + 1
        
        history = router.get_command_history(limit=1)
        assert len(history) == 1
        assert history[0]['command'] == '/test_command'
        
        router.clear_command_history()
        assert len(router.command_history) == 0


class TestToolRegistry:
    """Test the ToolRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create a ToolRegistry instance for testing."""
        return ToolRegistry()
    
    def test_tool_registry_initialization(self, registry):
        """Test ToolRegistry initialization."""
        assert registry is not None
        assert registry.unrestricted_mode is True
        assert registry.enable_shell_commands is True
        assert registry.enable_file_operations is True
        assert len(registry.tools) > 0  # Should have built-in tools
    
    def test_tool_registration(self, registry):
        """Test custom tool registration."""
        def test_tool(arg1, arg2):
            return f"test result: {arg1}, {arg2}"
        
        initial_count = len(registry.tools)
        registry.register_tool('test_tool', test_tool, 'Test tool for testing')
        
        assert len(registry.tools) == initial_count + 1
        assert 'test_tool' in registry.tools
        assert registry.tools['test_tool']['description'] == 'Test tool for testing'
    
    def test_tool_unregistration(self, registry):
        """Test tool unregistration."""
        def test_tool():
            return "test"
        
        registry.register_tool('temp_tool', test_tool)
        assert 'temp_tool' in registry.tools
        
        success = registry.unregister_tool('temp_tool')
        assert success is True
        assert 'temp_tool' not in registry.tools
        
        # Test unregistering non-existent tool
        success = registry.unregister_tool('nonexistent_tool')
        assert success is False
    
    @pytest.mark.asyncio
    async def test_tool_execution(self, registry):
        """Test tool execution."""
        def test_tool(message):
            return f"Tool executed with: {message}"
        
        registry.register_tool('test_exec_tool', test_tool)
        
        result = await registry.execute_tool('test_exec_tool', 'hello world')
        
        assert result['success'] is True
        assert 'Tool executed with: hello world' in result['result']
        assert result['tool_name'] == 'test_exec_tool'
    
    @pytest.mark.asyncio
    async def test_tool_execution_error(self, registry):
        """Test tool execution error handling."""
        def error_tool():
            raise ValueError("Test error")
        
        registry.register_tool('error_tool', error_tool)
        
        result = await registry.execute_tool('error_tool')
        
        assert result['success'] is False
        assert 'Test error' in result['error']
    
    @pytest.mark.asyncio
    async def test_nonexistent_tool_execution(self, registry):
        """Test execution of non-existent tool."""
        result = await registry.execute_tool('nonexistent_tool')
        
        assert result['success'] is False
        assert 'not found' in result['error']
        assert 'available_tools' in result
    
    def test_tool_list(self, registry):
        """Test getting tool list."""
        tools = registry.get_tool_list()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        # Check that built-in tools are present
        tool_names = [tool['name'] for tool in tools]
        assert 'run_shell_command' in tool_names
        assert 'read_file' in tool_names
        assert 'write_file' in tool_names
    
    @pytest.mark.asyncio
    async def test_shell_tool(self, registry):
        """Test the built-in shell command tool."""
        result = await registry.execute_tool('run_shell_command', 'echo test_shell')
        
        assert result['success'] is True
        assert 'test_shell' in result['result']['stdout']
    
    def test_file_tools(self, registry):
        """Test the built-in file operation tools."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('test content')
            temp_file = f.name
        
        try:
            # Test read_file
            result = registry._read_file(temp_file)
            assert 'content' in result
            assert result['content'] == 'test content'
            
            # Test write_file
            new_content = 'new test content'
            result = registry._write_file(temp_file, new_content)
            assert result['success'] is True
            
            # Verify the write
            result = registry._read_file(temp_file)
            assert result['content'] == new_content
            
        finally:
            os.unlink(temp_file)
    
    def test_directory_tool(self, registry):
        """Test the directory listing tool."""
        result = registry._list_directory('.')
        
        assert 'items' in result
        assert isinstance(result['items'], list)
        assert len(result['items']) > 0


class TestToolUseLogger:
    """Test the ToolUseLogger class."""
    
    @pytest.fixture
    def logger_instance(self):
        """Create a ToolUseLogger instance for testing."""
        return ToolUseLogger(db_path=':memory:')  # Use in-memory database
    
    def test_logger_initialization(self, logger_instance):
        """Test ToolUseLogger initialization."""
        assert logger_instance is not None
        assert logger_instance.db_path == ':memory:'
    
    def test_command_execution_logging(self, logger_instance):
        """Test logging of command executions."""
        command = '/run_shell "echo test"'
        result = {
            'success': True,
            'output': 'test output',
            'error': '',
            'exit_code': 0,
            'duration': 0.5
        }
        
        logger_instance.log_command_execution(command, 'test_executor', result)
        
        # Query the logs
        logs = logger_instance.query_logs(limit=1)
        assert len(logs) == 1
        assert logs[0]['command'] == command
        assert logs[0]['executor'] == 'test_executor'
        assert logs[0]['success'] is True
    
    def test_tool_usage_logging(self, logger_instance):
        """Test logging of tool usage."""
        logger_instance.log_tool_usage(
            tool_name='test_tool',
            executor='test_executor',
            tool_args='arg1 arg2',
            success=True,
            result='tool result',
            duration=0.3
        )
        
        logs = logger_instance.query_logs(log_type='tool', limit=1)
        assert len(logs) == 1
        assert logs[0]['tool_name'] == 'test_tool'
        assert logs[0]['success'] is True
    
    def test_agent_method_logging(self, logger_instance):
        """Test logging of agent method calls."""
        logger_instance.log_agent_method_call(
            agent_name='test_agent',
            method_name='test_method',
            executor='test_executor',
            arguments='test args',
            success=True,
            result='method result'
        )
        
        logs = logger_instance.query_logs(log_type='agent', limit=1)
        assert len(logs) == 1
        assert logs[0]['agent_name'] == 'test_agent'
        assert logs[0]['method_name'] == 'test_method'
    
    def test_system_event_logging(self, logger_instance):
        """Test logging of system events."""
        logger_instance.log_system_event(
            event_type='test_event',
            description='Test event description',
            data={'key': 'value'}
        )
        
        logs = logger_instance.query_logs(log_type='event', limit=1)
        assert len(logs) == 1
        assert logs[0]['event_type'] == 'test_event'
        assert logs[0]['description'] == 'Test event description'
    
    def test_log_querying(self, logger_instance):
        """Test log querying with search."""
        # Add some test logs
        logger_instance.log_command_execution('/run_shell "ls"', 'user', {'success': True})
        logger_instance.log_command_execution('/pip_install requests', 'system', {'success': False})
        
        # Query all logs
        all_logs = logger_instance.query_logs()
        assert len(all_logs) == 2
        
        # Query with search
        shell_logs = logger_instance.query_logs('run_shell')
        assert len(shell_logs) == 1
        assert 'run_shell' in shell_logs[0]['command']
        
        pip_logs = logger_instance.query_logs('pip_install')
        assert len(pip_logs) == 1
        assert 'pip_install' in pip_logs[0]['command']
    
    def test_execution_stats(self, logger_instance):
        """Test execution statistics."""
        # Add some test data
        logger_instance.log_command_execution('/test1', 'user', {'success': True})
        logger_instance.log_command_execution('/test2', 'user', {'success': False})
        logger_instance.log_tool_usage('tool1', 'user', '', True)
        
        stats = logger_instance.get_execution_stats(hours=24)
        
        assert stats['command_executions'] == 2
        assert stats['successful_commands'] == 1
        assert stats['failed_commands'] == 1
        assert stats['tool_usage'] == 1


class TestIntegration:
    """Integration tests for the complete Phase 13 system."""
    
    @pytest.mark.asyncio
    async def test_full_command_flow(self):
        """Test complete command flow from router to tool registry."""
        # Create instances
        tool_registry = ToolRegistry()
        agent_registry = AgentRegistry()
        router = CommandRouter(tool_registry, agent_registry)
        
        # Test shell command
        result = await router.execute_command('/run_shell "echo integration_test"', 'test')
        assert result['success'] is True
        assert 'integration_test' in result['output']
        
        # Test tool command
        result = await router.execute_command('/run_tool list_directory .', 'test')
        assert result['success'] is True
    
    def test_global_instances(self):
        """Test global instance getters."""
        router1 = get_command_router()
        router2 = get_command_router()
        assert router1 is router2  # Should be the same instance
        
        tool_reg1 = get_tool_registry()
        tool_reg2 = get_tool_registry()
        assert tool_reg1 is tool_reg2
        
        agent_reg1 = get_agent_registry()
        agent_reg2 = get_agent_registry()
        assert agent_reg1 is agent_reg2
    
    def test_synchronous_routing(self):
        """Test synchronous command routing wrapper."""
        result = route_command('/run_shell "echo sync_test"', 'test')
        assert result['success'] is True
        assert 'sync_test' in result['output']
    
    @pytest.mark.asyncio
    async def test_asynchronous_routing(self):
        """Test asynchronous command routing."""
        result = await route_command_async('/run_shell "echo async_test"', 'test')
        assert result['success'] is True
        assert 'async_test' in result['output']


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def router(self):
        """Create a CommandRouter instance for testing."""
        return CommandRouter()
    
    @pytest.mark.asyncio
    async def test_command_timeout(self, router):
        """Test command timeout handling."""
        # This test might be system-dependent
        command = '/run_shell "sleep 1"'  # Short sleep for testing
        result = await router.execute_command(command, 'test')
        # Should complete successfully if timeout is reasonable
        assert 'success' in result
    
    @pytest.mark.asyncio
    async def test_malformed_command(self, router):
        """Test handling of malformed commands."""
        result = await router.execute_command('malformed', 'test')
        assert result['success'] is False
        assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_empty_command(self, router):
        """Test handling of empty commands."""
        result = await router.execute_command('', 'test')
        assert result['success'] is False
    
    def test_restricted_mode_simulation(self):
        """Test behavior when unrestricted mode is disabled."""
        # Create router with restricted settings
        with patch('agents.command_router.get_config') as mock_config:
            mock_config.return_value = {
                'command_router': {
                    'unrestricted_mode': False,
                    'allow_all_shell_commands': False,
                    'enable_pip_install': False
                }
            }
            
            restricted_router = CommandRouter()
            assert restricted_router.unrestricted_mode is False
            assert restricted_router.allow_all_shell_commands is False
            assert restricted_router.enable_pip_install is False


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])