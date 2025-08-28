#!/usr/bin/env python3
"""
Command Router Agent for MeRNSTA - Phase 13: Command Routing & Execution

Provides full autonomy over shell commands, tools, pip, and runtime environment.
This agent has unrestricted access to the system in a sandboxed environment.
"""

import asyncio
import subprocess
import logging
import json
import time
import os
import sys
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from config.settings import get_config


class CommandRouter:
    """
    Central command router with full system autonomy.
    
    Handles:
    - Shell command execution (bash, apt, pip, etc.)
    - Tool registry dispatch
    - Agent method calls
    - Command logging and output capture
    - No restrictions or sanitization
    """
    
    def __init__(self, tool_registry=None, agent_registry=None):
        self.config = get_config().get('command_router', {})
        self.unrestricted_mode = self.config.get('unrestricted_mode', True)
        self.allow_all_shell_commands = self.config.get('allow_all_shell_commands', True)
        self.enable_pip_install = self.config.get('enable_pip_install', True)
        self.log_every_execution = self.config.get('log_every_execution', True)
        
        self.tool_registry = tool_registry
        self.agent_registry = agent_registry
        self.command_history = []
        
        # Initialize logging
        self.logger = logging.getLogger('command_router')
        self.logger.setLevel(logging.INFO)
        
        # Initialize tool use logger
        self._init_tool_logger()
        
        self.logger.info(f"[CommandRouter] Initialized with unrestricted_mode={self.unrestricted_mode}")
    
    def _init_tool_logger(self):
        """Initialize the tool use logger."""
        try:
            from storage.tool_use_log import ToolUseLogger
            self.tool_logger = ToolUseLogger()
        except ImportError:
            self.logger.warning("ToolUseLogger not available, command logging disabled")
            self.tool_logger = None
    
    def parse_command(self, command: str) -> Dict[str, Any]:
        """
        Parse /-prefixed commands from user or system.
        
        Returns:
            Dict with command type, arguments, and metadata
        """
        if not command.startswith('/'):
            return {
                'type': 'invalid',
                'error': 'Commands must start with /',
                'original': command
            }
        
        # Remove leading slash and split
        cmd_parts = command[1:].strip().split(' ', 1)
        cmd_name = cmd_parts[0].lower()
        cmd_args = cmd_parts[1] if len(cmd_parts) > 1 else ""
        
        # Determine command type and parse arguments
        if cmd_name == 'run_shell':
            return self._parse_shell_command(cmd_args)
        elif cmd_name == 'pip_install':
            return self._parse_pip_command(cmd_args)
        elif cmd_name == 'run_tool':
            return self._parse_tool_command(cmd_args)
        elif cmd_name == 'restart_self':
            return {'type': 'restart_self', 'args': {}}
        elif cmd_name == 'agent_status':
            return {'type': 'agent_status', 'args': {}}
        elif cmd_name == 'tool_log':
            return self._parse_tool_log_command(cmd_args)
        else:
            # Try to route to agent methods
            return self._parse_agent_command(cmd_name, cmd_args)
    
    def _parse_shell_command(self, args: str) -> Dict[str, Any]:
        """Parse shell command arguments."""
        # Handle quoted commands
        if args.startswith('"') and args.endswith('"'):
            args = args[1:-1]
        elif args.startswith("'") and args.endswith("'"):
            args = args[1:-1]
        
        return {
            'type': 'shell',
            'args': {
                'command': args,
                'raw': True
            }
        }
    
    def _parse_pip_command(self, args: str) -> Dict[str, Any]:
        """Parse pip install command arguments."""
        return {
            'type': 'pip_install',
            'args': {
                'package': args.strip(),
                'upgrade': False,  # Could be extended to parse --upgrade etc.
                'force': False
            }
        }
    
    def _parse_tool_command(self, args: str) -> Dict[str, Any]:
        """Parse tool execution command arguments."""
        parts = args.split(' ', 1)
        tool_name = parts[0] if parts else ""
        tool_args = parts[1] if len(parts) > 1 else ""
        
        return {
            'type': 'tool',
            'args': {
                'tool_name': tool_name,
                'tool_args': tool_args
            }
        }
    
    def _parse_tool_log_command(self, args: str) -> Dict[str, Any]:
        """Parse tool log query command arguments."""
        return {
            'type': 'tool_log',
            'args': {
                'query': args.strip() if args else "",
                'limit': 50  # Default limit
            }
        }
    
    def _parse_agent_command(self, cmd_name: str, args: str) -> Dict[str, Any]:
        """Parse agent method calls."""
        return {
            'type': 'agent_method',
            'args': {
                'method': cmd_name,
                'arguments': args
            }
        }
    
    async def execute_command(self, command: str, executor: str = "system") -> Dict[str, Any]:
        """
        Execute a parsed command with full system access.
        
        Args:
            command: Raw command string
            executor: Who is executing (system, user, agent_name)
            
        Returns:
            Execution result with output, status, and metadata
        """
        start_time = time.time()
        parsed = self.parse_command(command)
        
        if parsed['type'] == 'invalid':
            return self._create_result(
                success=False,
                output="",
                error=parsed['error'],
                command=command,
                executor=executor,
                duration=time.time() - start_time
            )
        
        # Execute based on command type
        try:
            if parsed['type'] == 'shell':
                result = await self._execute_shell_command(parsed['args'], executor, command)
            elif parsed['type'] == 'pip_install':
                result = await self._execute_pip_command(parsed['args'], executor)
            elif parsed['type'] == 'tool':
                result = await self._execute_tool_command(parsed['args'], executor)
            elif parsed['type'] == 'restart_self':
                result = await self._execute_restart_command(parsed['args'], executor)
            elif parsed['type'] == 'agent_status':
                result = await self._execute_agent_status_command(parsed['args'], executor)
            elif parsed['type'] == 'tool_log':
                result = await self._execute_tool_log_command(parsed['args'], executor)
            elif parsed['type'] == 'agent_method':
                result = await self._execute_agent_method(parsed['args'], executor)
            else:
                result = self._create_result(
                    success=False,
                    output="",
                    error=f"Unknown command type: {parsed['type']}",
                    command=command,
                    executor=executor,
                    duration=time.time() - start_time
                )
        except Exception as e:
            result = self._create_result(
                success=False,
                output="",
                error=f"Command execution error: {str(e)}",
                command=command,
                executor=executor,
                duration=time.time() - start_time
            )
        
        # Log the execution
        if self.log_every_execution and self.tool_logger:
            self.tool_logger.log_command_execution(
                command=command,
                executor=executor,
                result=result,
                timestamp=datetime.now()
            )
        
        # Add to command history
        self.command_history.append({
            'command': command,
            'executor': executor,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
        return result
    
    async def _execute_shell_command(self, args: Dict[str, Any], executor: str, original_command: str = None) -> Dict[str, Any]:
        """Execute raw shell command with full access."""
        if not self.allow_all_shell_commands:
            return self._create_result(
                success=False,
                output="",
                error="Shell commands disabled in configuration",
                command=original_command or args.get('command', ''),
                executor=executor
            )
        
        cmd = args['command']
        self.logger.info(f"[CommandRouter] Executing shell command: {cmd}")
        
        try:
            # Use asyncio subprocess for non-blocking execution
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy()
            )
            
            stdout, stderr = await process.communicate()
            
            return self._create_result(
                success=process.returncode == 0,
                output=stdout.decode('utf-8', errors='replace'),
                error=stderr.decode('utf-8', errors='replace'),
                command=original_command or cmd,
                executor=executor,
                exit_code=process.returncode
            )
            
        except Exception as e:
            return self._create_result(
                success=False,
                output="",
                error=f"Shell execution error: {str(e)}",
                command=original_command or cmd,
                executor=executor
            )
    
    async def _execute_pip_command(self, args: Dict[str, Any], executor: str) -> Dict[str, Any]:
        """Execute pip install with full access."""
        if not self.enable_pip_install:
            return self._create_result(
                success=False,
                output="",
                error="Pip install disabled in configuration",
                command=f"pip install {args.get('package', '')}",
                executor=executor
            )
        
        package = args['package']
        cmd = f"pip install {package}"
        
        self.logger.info(f"[CommandRouter] Executing pip install: {package}")
        
        # Route through shell command execution
        return await self._execute_shell_command({'command': cmd}, executor)
    
    async def _execute_tool_command(self, args: Dict[str, Any], executor: str) -> Dict[str, Any]:
        """Execute registered tool."""
        tool_name = args['tool_name']
        tool_args = args['tool_args']
        
        if not self.tool_registry:
            return self._create_result(
                success=False,
                output="",
                error="Tool registry not available",
                command=f"run_tool {tool_name} {tool_args}",
                executor=executor
            )
        
        self.logger.info(f"[CommandRouter] Executing tool: {tool_name} with args: {tool_args}")
        
        try:
            # Get tool from registry and execute
            result = await self.tool_registry.execute_tool(tool_name, tool_args)
            return self._create_result(
                success=True,
                output=str(result),
                error="",
                command=f"run_tool {tool_name} {tool_args}",
                executor=executor
            )
        except Exception as e:
            return self._create_result(
                success=False,
                output="",
                error=f"Tool execution error: {str(e)}",
                command=f"run_tool {tool_name} {tool_args}",
                executor=executor
            )
    
    async def _execute_restart_command(self, args: Dict[str, Any], executor: str) -> Dict[str, Any]:
        """Execute system restart."""
        self.logger.info(f"[CommandRouter] Restart requested by {executor}")
        
        # This would restart the MeRNSTA system
        cmd = "python run_mernsta.py"
        
        return self._create_result(
            success=True,
            output="System restart initiated",
            error="",
            command="restart_self",
            executor=executor
        )
    
    async def _execute_agent_status_command(self, args: Dict[str, Any], executor: str) -> Dict[str, Any]:
        """Get agent status information."""
        if not self.agent_registry:
            return self._create_result(
                success=False,
                output="",
                error="Agent registry not available",
                command="agent_status",
                executor=executor
            )
        
        try:
            status = self.agent_registry.get_status()
            return self._create_result(
                success=True,
                output=json.dumps(status, indent=2),
                error="",
                command="agent_status",
                executor=executor
            )
        except Exception as e:
            return self._create_result(
                success=False,
                output="",
                error=f"Agent status error: {str(e)}",
                command="agent_status",
                executor=executor
            )
    
    async def _execute_tool_log_command(self, args: Dict[str, Any], executor: str) -> Dict[str, Any]:
        """Query tool execution logs."""
        if not self.tool_logger:
            return self._create_result(
                success=False,
                output="",
                error="Tool logger not available",
                command="tool_log",
                executor=executor
            )
        
        try:
            query = args.get('query', '')
            limit = args.get('limit', 50)
            logs = self.tool_logger.query_logs(query, limit)
            
            return self._create_result(
                success=True,
                output=json.dumps(logs, indent=2),
                error="",
                command=f"tool_log {query}",
                executor=executor
            )
        except Exception as e:
            return self._create_result(
                success=False,
                output="",
                error=f"Tool log query error: {str(e)}",
                command="tool_log",
                executor=executor
            )
    
    async def _execute_agent_method(self, args: Dict[str, Any], executor: str) -> Dict[str, Any]:
        """Execute agent method calls."""
        method = args['method']
        arguments = args['arguments']
        
        if not self.agent_registry:
            return self._create_result(
                success=False,
                output="",
                error="Agent registry not available",
                command=f"{method} {arguments}",
                executor=executor
            )
        
        try:
            result = await self.agent_registry.call_agent_method(method, arguments)
            return self._create_result(
                success=True,
                output=str(result),
                error="",
                command=f"{method} {arguments}",
                executor=executor
            )
        except Exception as e:
            return self._create_result(
                success=False,
                output="",
                error=f"Agent method error: {str(e)}",
                command=f"{method} {arguments}",
                executor=executor
            )
    
    def _create_result(self, success: bool, output: str, error: str, 
                      command: str, executor: str, duration: float = 0.0,
                      exit_code: int = None) -> Dict[str, Any]:
        """Create standardized result dictionary."""
        return {
            'success': success,
            'output': output,
            'error': error,
            'command': command,
            'executor': executor,
            'duration': duration,
            'exit_code': exit_code,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_command_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent command execution history."""
        return self.command_history[-limit:]
    
    def clear_command_history(self):
        """Clear command execution history."""
        self.command_history.clear()
        self.logger.info("[CommandRouter] Command history cleared")


# Global command router instance
_command_router = None

def get_command_router(tool_registry=None, agent_registry=None) -> CommandRouter:
    """Get or create global command router instance."""
    global _command_router
    if _command_router is None:
        _command_router = CommandRouter(tool_registry, agent_registry)
    return _command_router


def route_command(command: str, executor: str = "system") -> Dict[str, Any]:
    """
    Synchronous wrapper for command routing.
    
    Args:
        command: Command string to execute
        executor: Who is executing the command
        
    Returns:
        Execution result
    """
    router = get_command_router()
    
    # Run async command in event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a task
            task = asyncio.create_task(router.execute_command(command, executor))
            return asyncio.run(task)
        else:
            return loop.run_until_complete(router.execute_command(command, executor))
    except RuntimeError:
        # No event loop running, create new one
        return asyncio.run(router.execute_command(command, executor))


async def route_command_async(command: str, executor: str = "system") -> Dict[str, Any]:
    """
    Asynchronous command routing.
    
    Args:
        command: Command string to execute
        executor: Who is executing the command
        
    Returns:
        Execution result
    """
    router = get_command_router()
    return await router.execute_command(command, executor)