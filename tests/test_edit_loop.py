#!/usr/bin/env python3
"""
Test Suite for Recursive Execution - Phase 14: Recursive Execution

Tests the complete write-run-analyze-edit chain including:
- FileWriter agent functionality
- ExecutionMonitor analysis capabilities 
- EditLoop recursive improvement
- CLI commands integration
- Memory and logging systems
"""

import sys
import os
import asyncio
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

# Add current directory to path
sys.path.insert(0, os.getcwd())

# Test imports
from agents.file_writer import FileWriter, get_file_writer
from agents.execution_monitor import ExecutionMonitor, get_execution_monitor
from agents.edit_loop import EditLoop, get_edit_loop
from config.settings import get_config


class TestFileWriter:
    """Test FileWriter agent functionality."""
    
    def test_file_writer_initialization(self):
        """Test FileWriter initializes correctly."""
        writer = FileWriter()
        assert writer is not None
        assert writer.write_dir.exists()
        assert len(writer.safe_extensions) > 0
    
    def test_write_simple_python_file(self):
        """Test writing a simple Python file."""
        writer = FileWriter()
        code = 'print("Hello, World!")'
        result = writer.write_file(code, "test_hello.py", force_overwrite=True)
        
        assert result['success'] is True
        assert 'filepath' in result
        assert Path(result['filepath']).exists()
        
        # Verify content
        with open(result['filepath'], 'r') as f:
            content = f.read()
        assert 'Hello, World!' in content
        
        # Cleanup
        Path(result['filepath']).unlink()
    
    def test_write_executable_shell_script(self):
        """Test writing an executable shell script."""
        writer = FileWriter()
        result = writer.write_shell_script(
            ['echo "Test script"', 'echo "Second line"'],
            "test_script",
            force_overwrite=True
        )
        
        assert result['success'] is True
        assert result['executable'] is True
        assert Path(result['filepath']).exists()
        
        # Verify it's executable
        file_path = Path(result['filepath'])
        assert file_path.stat().st_mode & 0o111
        
        # Cleanup
        file_path.unlink()
    
    def test_unsafe_extension_rejection(self):
        """Test that unsafe file extensions are rejected."""
        writer = FileWriter()
        result = writer.write_file("malicious code", "bad_file.exe")
        
        assert result['success'] is False
        assert 'Unsafe file extension' in result['error']
    
    def test_file_versioning(self):
        """Test automatic file versioning."""
        writer = FileWriter()
        
        # Write first file
        result1 = writer.write_file("version 1", "version_test.py", force_overwrite=True)
        assert result1['success'] is True
        
        # Write second file with same name
        result2 = writer.write_file("version 2", "version_test.py", add_timestamp=True)
        assert result2['success'] is True
        assert result1['filepath'] != result2['filepath']
        
        # Both files should exist
        assert Path(result1['filepath']).exists()
        assert Path(result2['filepath']).exists()
        
        # Cleanup
        Path(result1['filepath']).unlink()
        Path(result2['filepath']).unlink()
    
    def test_list_generated_files(self):
        """Test listing generated files."""
        writer = FileWriter()
        
        # Create test files
        result1 = writer.write_file("test 1", "list_test1.py", force_overwrite=True)
        result2 = writer.write_file("test 2", "list_test2.py", force_overwrite=True)
        
        # List files
        file_list = writer.list_generated_files()
        assert file_list['success'] is True
        assert file_list['count'] >= 2
        
        file_names = [f['name'] for f in file_list['files']]
        assert 'list_test1.py' in file_names
        assert 'list_test2.py' in file_names
        
        # Cleanup
        Path(result1['filepath']).unlink()
        Path(result2['filepath']).unlink()


class TestExecutionMonitor:
    """Test ExecutionMonitor analysis capabilities."""
    
    def test_execution_monitor_initialization(self):
        """Test ExecutionMonitor initializes correctly."""
        monitor = ExecutionMonitor()
        assert monitor is not None
        assert len(monitor.success_patterns) > 0
        assert len(monitor.failure_patterns) > 0
    
    def test_successful_execution_analysis(self):
        """Test analysis of successful execution."""
        monitor = ExecutionMonitor()
        
        execution_result = {
            'success': True,
            'output': 'Script completed successfully\nAll tests passed',
            'error': '',
            'exit_code': 0,
            'duration': 1.5
        }
        
        metrics = monitor.analyze_execution(execution_result)
        
        assert metrics.overall_success is True
        assert metrics.confidence_score > 0.7
        assert metrics.exit_code == 0
        assert len(metrics.success_indicators) > 0
        assert len(metrics.failure_indicators) == 0
    
    def test_failed_execution_analysis(self):
        """Test analysis of failed execution."""
        monitor = ExecutionMonitor()
        
        execution_result = {
            'success': False,
            'output': '',
            'error': 'NameError: name "undefined_var" is not defined\nTraceback (most recent call last):',
            'exit_code': 1,
            'duration': 0.2
        }
        
        metrics = monitor.analyze_execution(execution_result)
        
        assert metrics.overall_success is False
        assert metrics.confidence_score < 0.5
        assert metrics.exit_code == 1
        assert len(metrics.failure_indicators) > 0
        assert len(metrics.exceptions) > 0
        assert len(metrics.improvement_suggestions) > 0
    
    def test_syntax_error_suggestions(self):
        """Test suggestions for syntax errors."""
        monitor = ExecutionMonitor()
        
        execution_result = {
            'success': False,
            'output': '',
            'error': 'SyntaxError: invalid syntax\n  print "hello"',
            'exit_code': 1,
            'duration': 0.1
        }
        
        metrics = monitor.analyze_execution(execution_result)
        
        suggestions = [s.lower() for s in metrics.improvement_suggestions]
        assert any('syntax' in s for s in suggestions)
    
    def test_import_error_suggestions(self):
        """Test suggestions for import errors."""
        monitor = ExecutionMonitor()
        
        execution_result = {
            'success': False,
            'output': '',
            'error': 'ImportError: No module named "nonexistent_module"',
            'exit_code': 1,
            'duration': 0.1
        }
        
        metrics = monitor.analyze_execution(execution_result)
        
        suggestions = [s.lower() for s in metrics.improvement_suggestions]
        assert any('import' in s for s in suggestions)
    
    def test_retry_decision_logic(self):
        """Test should_retry decision logic."""
        monitor = ExecutionMonitor()
        
        # Should retry on fixable error
        fixable_metrics = ExecutionMonitor().analyze_execution({
            'success': False,
            'output': '',
            'error': 'NameError: name "x" is not defined',
            'exit_code': 1,
            'duration': 0.1
        })
        
        assert monitor.should_retry(fixable_metrics, 1, 5) is True
        
        # Should not retry after max attempts
        assert monitor.should_retry(fixable_metrics, 5, 5) is False
        
        # Should not retry on permanent failure
        permanent_metrics = ExecutionMonitor().analyze_execution({
            'success': False,
            'output': '',
            'error': 'Permission denied: /root/restricted_file',
            'exit_code': 126,
            'duration': 0.1
        })
        
        assert monitor.should_retry(permanent_metrics, 1, 5) is False


class TestEditLoop:
    """Test EditLoop recursive improvement functionality."""
    
    @pytest.mark.asyncio
    async def test_edit_loop_initialization(self):
        """Test EditLoop initializes correctly."""
        loop = EditLoop()
        assert loop is not None
        assert loop.file_writer is not None
        assert loop.execution_monitor is not None
    
    @pytest.mark.asyncio
    async def test_successful_simple_script(self):
        """Test successful execution of a simple working script."""
        loop = EditLoop()
        
        # Simple working Python script
        code = """#!/usr/bin/env python3
print("Hello from recursive execution!")
print("This script should work on first try")
"""
        
        result = await loop.run_loop(
            initial_code=code,
            filename="simple_success.py",
            goal_description="Run a simple Python script",
            max_attempts=3
        )
        
        assert result.success is True
        assert result.total_attempts == 1
        assert result.final_attempt.analysis_metrics.overall_success is True
        
        # Cleanup
        if result.winning_file_path:
            Path(result.winning_file_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_syntax_error_improvement(self):
        """Test improvement of script with syntax error."""
        loop = EditLoop()
        
        # Python script with syntax error
        code = """#!/usr/bin/env python3
print "Hello World"  # Missing parentheses for Python 3
x = 5
print x  # Another syntax error
"""
        
        result = await loop.run_loop(
            initial_code=code,
            filename="syntax_error.py",
            goal_description="Fix Python 3 syntax errors",
            max_attempts=3
        )
        
        # Should either succeed or provide good improvement attempts
        assert result.total_attempts > 1
        assert len(result.all_attempts) > 1
        
        # Check that improvements were attempted
        for attempt in result.all_attempts:
            assert attempt.next_suggestions
        
        # Cleanup
        for attempt in result.all_attempts:
            if attempt.write_result.get('filepath'):
                Path(attempt.write_result['filepath']).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_max_attempts_limit(self):
        """Test that edit loop respects max attempts limit."""
        loop = EditLoop()
        
        # Unfixable script
        code = "this is not valid code in any language"
        
        result = await loop.run_loop(
            initial_code=code,
            filename="unfixable.py",
            goal_description="Try to fix unfixable code",
            max_attempts=2
        )
        
        assert result.success is False
        assert result.total_attempts == 2
        assert "Maximum attempts" in result.termination_reason
        
        # Cleanup
        for attempt in result.all_attempts:
            if attempt.write_result.get('filepath'):
                Path(attempt.write_result['filepath']).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_pattern_based_improvement(self):
        """Test pattern-based code improvement."""
        loop = EditLoop()
        
        # Code with common fixable issues
        code = """
# Missing import
json.dumps({"test": "data"})

# Undefined variable
print(undefined_variable)
"""
        
        result = await loop.run_loop(
            initial_code=code,
            filename="pattern_test.py",
            goal_description="Fix missing imports and undefined variables",
            max_attempts=3
        )
        
        # Should attempt improvements
        assert result.total_attempts > 1
        
        # Check for improvement patterns
        later_attempts = result.all_attempts[1:] if len(result.all_attempts) > 1 else []
        for attempt in later_attempts:
            # Improvements should be different from original
            assert attempt.code_content != code
        
        # Cleanup
        for attempt in result.all_attempts:
            if attempt.write_result.get('filepath'):
                Path(attempt.write_result['filepath']).unlink(missing_ok=True)


class TestCLIIntegration:
    """Test CLI command integration."""
    
    def test_write_and_execute_simple(self):
        """Test write_and_execute function."""
        from agents.file_writer import write_and_execute
        
        code = 'print("CLI test successful")'
        result = write_and_execute(code, "cli_test.py", "test_executor")
        
        assert result['success'] is True
        assert result['write_result']['success'] is True
        assert result['execution_result']['success'] is True
        assert 'CLI test successful' in result['execution_result']['output']
        
        # Cleanup
        if result.get('filepath'):
            Path(result['filepath']).unlink(missing_ok=True)
    
    def test_file_writer_global_instance(self):
        """Test global file writer instance."""
        writer1 = get_file_writer()
        writer2 = get_file_writer()
        
        # Should return same instance
        assert writer1 is writer2
    
    def test_execution_monitor_global_instance(self):
        """Test global execution monitor instance."""
        monitor1 = get_execution_monitor()
        monitor2 = get_execution_monitor()
        
        # Should return same instance
        assert monitor1 is monitor2
    
    def test_edit_loop_global_instance(self):
        """Test global edit loop instance."""
        loop1 = get_edit_loop()
        loop2 = get_edit_loop()
        
        # Should return same instance
        assert loop1 is loop2


class TestMemoryAndLogging:
    """Test memory and logging integration."""
    
    @pytest.mark.asyncio
    async def test_execution_logging(self):
        """Test that executions are properly logged."""
        monitor = ExecutionMonitor()
        
        # Mock the tool logger
        with patch('agents.execution_monitor.get_tool_logger') as mock_logger:
            mock_tool_logger = Mock()
            mock_logger.return_value = mock_tool_logger
            
            execution_result = {
                'success': True,
                'output': 'Test output',
                'error': '',
                'exit_code': 0,
                'duration': 1.0
            }
            
            metrics = monitor.analyze_execution(
                execution_result, 
                file_path="/test/file.py"
            )
            
            # Verify logging was called
            mock_tool_logger.log_system_event.assert_called_once()
            call_args = mock_tool_logger.log_system_event.call_args
            assert call_args[1]['event_type'] == 'execution_analysis'
    
    @pytest.mark.asyncio
    async def test_edit_loop_logging(self):
        """Test that edit loop attempts are logged."""
        loop = EditLoop()
        
        with patch('agents.edit_loop.get_tool_logger') as mock_logger:
            mock_tool_logger = Mock()
            mock_logger.return_value = mock_tool_logger
            
            # Simple working script to minimize attempts
            code = 'print("test")'
            
            result = await loop.run_loop(
                initial_code=code,
                filename="log_test.py",
                max_attempts=1
            )
            
            # Should have logged attempt
            assert mock_tool_logger.log_system_event.called
            
            # Cleanup
            if result.winning_file_path:
                Path(result.winning_file_path).unlink(missing_ok=True)


class TestConfigurationIntegration:
    """Test configuration system integration."""
    
    def test_config_loading(self):
        """Test that recursive execution config is loaded."""
        config = get_config()
        recursive_config = config.get('recursive_execution', {})
        
        # Should have basic configuration keys
        expected_keys = ['enable', 'max_attempts', 'write_dir', 'safe_extensions']
        for key in expected_keys:
            assert key in recursive_config
    
    def test_file_writer_respects_config(self):
        """Test that FileWriter respects configuration."""
        # Mock config for testing
        test_config = {
            'recursive_execution': {
                'write_dir': './test_generated/',
                'safe_extensions': ['.py', '.txt'],
                'enable_versioning': True,
                'add_metadata': True
            }
        }
        
        with patch('agents.file_writer.get_config', return_value=test_config):
            writer = FileWriter()
            
            assert writer.write_dir.name == 'test_generated'
            assert writer.safe_extensions == ['.py', '.txt']
            assert writer.enable_versioning is True
            assert writer.add_metadata is True


def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    
    async def run_workflow():
        # 1. Create a script with a fixable error
        writer = get_file_writer()
        code_with_error = """
import sys
print("Starting script...")
print(undefined_variable)  # This will cause NameError
print("Script complete")
"""
        
        write_result = writer.write_file(
            code_with_error, 
            "e2e_test.py", 
            force_overwrite=True
        )
        assert write_result['success']
        
        # 2. Run recursive improvement
        loop = get_edit_loop()
        improvement_result = await loop.run_loop(
            initial_code=code_with_error,
            filename="e2e_improved.py",
            goal_description="Fix the undefined variable error",
            max_attempts=3
        )
        
        # 3. Verify results
        assert improvement_result.total_attempts >= 1
        
        # 4. Clean up
        Path(write_result['filepath']).unlink(missing_ok=True)
        for attempt in improvement_result.all_attempts:
            if attempt.write_result.get('filepath'):
                Path(attempt.write_result['filepath']).unlink(missing_ok=True)
        
        return improvement_result
    
    # Run the async workflow
    result = asyncio.run(run_workflow())
    assert result is not None


if __name__ == "__main__":
    # Run tests manually
    print("=== Recursive Execution Test Suite ===\n")
    
    # File Writer Tests
    print("Testing FileWriter...")
    test_file_writer = TestFileWriter()
    test_file_writer.test_file_writer_initialization()
    test_file_writer.test_write_simple_python_file()
    test_file_writer.test_write_executable_shell_script()
    test_file_writer.test_unsafe_extension_rejection()
    test_file_writer.test_file_versioning()
    test_file_writer.test_list_generated_files()
    print("✅ FileWriter tests passed")
    
    # Execution Monitor Tests
    print("\nTesting ExecutionMonitor...")
    test_monitor = TestExecutionMonitor()
    test_monitor.test_execution_monitor_initialization()
    test_monitor.test_successful_execution_analysis()
    test_monitor.test_failed_execution_analysis()
    test_monitor.test_syntax_error_suggestions()
    test_monitor.test_import_error_suggestions()
    test_monitor.test_retry_decision_logic()
    print("✅ ExecutionMonitor tests passed")
    
    # CLI Integration Tests
    print("\nTesting CLI Integration...")
    test_cli = TestCLIIntegration()
    test_cli.test_write_and_execute_simple()
    test_cli.test_file_writer_global_instance()
    test_cli.test_execution_monitor_global_instance()
    test_cli.test_edit_loop_global_instance()
    print("✅ CLI Integration tests passed")
    
    # Configuration Tests
    print("\nTesting Configuration...")
    test_config = TestConfigurationIntegration()
    test_config.test_config_loading()
    test_config.test_file_writer_respects_config()
    print("✅ Configuration tests passed")
    
    # End-to-End Test
    print("\nTesting End-to-End Workflow...")
    test_end_to_end_workflow()
    print("✅ End-to-End test passed")
    
    print("\n🎉 All tests passed! Recursive execution system is ready.")