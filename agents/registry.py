#!/usr/bin/env python3
"""
Agent & Tool Registry for MeRNSTA Multi-Agent Cognitive System

Centralized registry for loading, managing, and accessing all cognitive agents and tools.
Phase 13: Added full shell command and tool execution capabilities.
"""

import asyncio
import subprocess
import logging
import json
import time
import os
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from config.settings import get_config

# Import all agent classes
from .planner import PlannerAgent
from .critic import CriticAgent
from .debater import DebaterAgent
from .recursive_planner import RecursivePlanner
from .self_prompter import SelfPromptGenerator
from .self_healer import SelfHealer
from .reflector import ReflectorAgent
from .architect_analyzer import ArchitectAnalyzer
from .code_refactorer import CodeRefactorer
from .world_modeler import WorldModeler
from .constraint_engine import ConstraintEngine
from .evolution_tree import SelfReplicator
from .self_replicator import AgentReplicator
from .upgrade_manager import UpgradeManager
from .decision_planner import DecisionPlanner
from .strategy_evaluator import StrategyEvaluator
from .task_selector import TaskSelector
from .debate_engine import DebateEngine
from .reflection_orchestrator import ReflectionOrchestrator
from .fast_reflex_agent import FastReflexAgent
from .mesh_manager import AgentMeshManager
from .meta_self_agent import MetaSelfAgent
from .cognitive_arbiter import CognitiveArbiter

class AgentRegistry:
    """
    Central registry for managing all cognitive agents.
    
    Provides:
    - Agent initialization and registration
    - Agent discovery and access
    - Configuration management
    - Health monitoring
    """
    
    def __init__(self):
        self.config = get_config().get('multi_agent', {})
        self.enabled = self.config.get('enabled', True)
        self.agents: Dict[str, Any] = {}
        self.agent_classes = {
            'planner': PlannerAgent,
            'critic': CriticAgent,
            'debater': DebaterAgent,
            'reflector': ReflectorAgent,
            'recursive_planner': RecursivePlanner,
            'self_prompter': SelfPromptGenerator,
            'self_healer': SelfHealer,
            'architect_analyzer': ArchitectAnalyzer,
            'code_refactorer': CodeRefactorer,
            'upgrade_manager': UpgradeManager,
            'world_modeler': WorldModeler,
            'constraint_engine': ConstraintEngine,
            'self_replicator': SelfReplicator,
            'agent_replicator': AgentReplicator,
            'decision_planner': DecisionPlanner,
            'strategy_evaluator': StrategyEvaluator,
            'task_selector': TaskSelector,
            'debate_engine': DebateEngine,
            'reflection_orchestrator': ReflectionOrchestrator,
            'fast_reflex': FastReflexAgent,
            'mesh_manager': AgentMeshManager,
            'meta_self': MetaSelfAgent,
            'cognitive_arbiter': CognitiveArbiter
        }
        
        if self.enabled:
            self._initialize_agents()
        else:
            logging.info("[AgentRegistry] Multi-agent system disabled in config")
    
    def _initialize_agents(self):
        """Initialize all enabled agents."""
        enabled_agents = self.config.get('agents', ['planner', 'critic', 'debater', 'reflector', 'recursive_planner', 'self_prompter', 'self_healer', 'architect_analyzer', 'code_refactorer', 'upgrade_manager', 'world_modeler', 'constraint_engine', 'self_replicator', 'agent_replicator'])
        
        for agent_name in enabled_agents:
            if agent_name in self.agent_classes:
                try:
                    agent_class = self.agent_classes[agent_name]
                    agent_instance = agent_class()
                    self.agents[agent_name] = agent_instance
                    logging.info(f"[AgentRegistry] Initialized {agent_name}Agent")
                except Exception as e:
                    logging.error(f"[AgentRegistry] Failed to initialize {agent_name}Agent: {e}")
            else:
                logging.warning(f"[AgentRegistry] Unknown agent type: {agent_name}")
        
        logging.info(f"[AgentRegistry] Initialized {len(self.agents)} agents: {list(self.agents.keys())}")
    
    def get_agent(self, name: str) -> Optional[Any]:
        """
        Get an agent by name.
        
        Args:
            name: Agent name (e.g., 'planner', 'critic')
            
        Returns:
            Agent instance or None if not found
        """
        return self.agents.get(name)
    
    def get_all_agents(self) -> Dict[str, Any]:
        """Get all registered agents."""
        return self.agents.copy()
    
    def get_agent_names(self) -> List[str]:
        """Get list of all registered agent names."""
        return list(self.agents.keys())
    
    def is_agent_available(self, name: str) -> bool:
        """Check if an agent is available."""
        return name in self.agents
    
    def execute_debate_mode(self, message: str, context: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        Execute debate mode - get responses from all agents.
        
        Args:
            message: The input message
            context: Additional context
            
        Returns:
            List of agent responses
        """
        if not self.config.get('debate_mode', True):
            return []
            
        results = []
        for agent_name, agent in self.agents.items():
            try:
                response = agent.respond(message, context)
                results.append({
                    'agent': agent_name,
                    'response': response
                })
            except Exception as e:
                logging.error(f"[AgentRegistry] Error in debate mode for {agent_name}: {e}")
                results.append({
                    'agent': agent_name,
                    'response': f"[{agent_name}Agent] Error: {str(e)}"
                })
        
        return results
    
    def get_agent_capabilities(self, name: str = None) -> Dict[str, Any]:
        """
        Get capabilities for a specific agent or all agents.
        
        Args:
            name: Agent name, or None for all agents
            
        Returns:
            Agent capabilities information
        """
        if name:
            agent = self.get_agent(name)
            if agent:
                return agent.get_capabilities()
            else:
                return {"error": f"Agent '{name}' not found"}
        else:
            return {
                agent_name: agent.get_capabilities()
                for agent_name, agent in self.agents.items()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and health."""
        return {
            "enabled": self.enabled,
            "total_agents": len(self.agents),
            "agent_names": list(self.agents.keys()),
            "debate_mode": self.config.get('debate_mode', True),
            "config": self.config,
            "agent_health": {
                name: {
                    "initialized": True,
                    "enabled": agent.enabled,
                    "has_llm": agent.llm_fallback is not None,
                    "has_symbolic": agent.symbolic_engine is not None,
                    "has_memory": agent.memory_system is not None
                }
                for name, agent in self.agents.items()
            }
        }
    
    def reload_agents(self):
        """Reload all agents (useful for config changes)."""
        logging.info("[AgentRegistry] Reloading agents...")
        self.agents.clear()
        
        # Reload config
        self.config = get_config().get('multi_agent', {})
        self.enabled = self.config.get('enabled', True)
        
        if self.enabled:
            self._initialize_agents()
        else:
            logging.info("[AgentRegistry] Multi-agent system disabled after reload")
    
    def add_custom_agent(self, name: str, agent_instance: Any):
        """
        Add a custom agent to the registry.
        
        Args:
            name: Agent name
            agent_instance: Agent instance
        """
        self.agents[name] = agent_instance
        logging.info(f"[AgentRegistry] Added custom agent: {name}")
    
    def remove_agent(self, name: str) -> bool:
        """
        Remove an agent from the registry.
        
        Args:
            name: Agent name
            
        Returns:
            True if agent was removed, False if not found
        """
        if name in self.agents:
            del self.agents[name]
            logging.info(f"[AgentRegistry] Removed agent: {name}")
            return True
        return False
    
    def __len__(self):
        """Return number of registered agents."""
        return len(self.agents)
    
    def __contains__(self, name: str):
        """Check if agent name is in registry."""
        return name in self.agents
    
    def __iter__(self):
        """Iterate over agent names."""
        return iter(self.agents)
    
    async def call_agent_method(self, method: str, arguments: str) -> Any:
        """
        Call a method on an agent or execute an agent command.
        
        Args:
            method: Method name or agent name
            arguments: Method arguments
            
        Returns:
            Method result
        """
        # Try to find agent by name first
        if method in self.agents:
            agent = self.agents[method]
            if hasattr(agent, 'respond'):
                return agent.respond(arguments)
            else:
                return f"Agent {method} does not have respond method"
        
        # Try to find method across all agents
        for agent_name, agent in self.agents.items():
            if hasattr(agent, method):
                method_func = getattr(agent, method)
                if callable(method_func):
                    try:
                        if arguments:
                            # Try to parse arguments (simple string parsing)
                            return method_func(arguments)
                        else:
                            return method_func()
                    except Exception as e:
                        return f"Error calling {agent_name}.{method}: {str(e)}"
        
        return f"Method '{method}' not found in any agent"
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status of all agents and the registry."""
        return self.get_system_status()

# Global registry instance
_registry = None

def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry instance."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry

def reload_agent_registry():
    """Reload the global agent registry."""
    global _registry
    if _registry:
        _registry.reload_agents()
    else:
        _registry = AgentRegistry()

# Convenience access to the global registry
AGENT_REGISTRY = get_agent_registry().get_all_agents()


class ToolRegistry:
    """
    Tool registry with full shell command and system access capabilities.
    
    Features:
    - Shell command execution (bash, apt, pip, etc.)
    - File operations (read, write, execute)
    - Background process management
    - Streaming stdout/stderr
    - Exit code capture
    - No restrictions in unrestricted mode
    """
    
    def __init__(self):
        self.config = get_config().get('tool_registry', {})
        self.unrestricted_mode = self.config.get('unrestricted_mode', True)
        self.enable_shell_commands = self.config.get('enable_shell_commands', True)
        self.enable_file_operations = self.config.get('enable_file_operations', True)
        self.enable_background_processes = self.config.get('enable_background_processes', True)
        
        self.tools = {}
        self.background_processes = {}
        
        # Initialize logging
        self.logger = logging.getLogger('tool_registry')
        self.logger.setLevel(logging.INFO)
        
        # Initialize tool use logger
        self._init_tool_logger()
        
        # Register built-in tools
        self._register_builtin_tools()
        
        self.logger.info(f"[ToolRegistry] Initialized with unrestricted_mode={self.unrestricted_mode}")
    
    def _init_tool_logger(self):
        """Initialize the tool use logger."""
        try:
            from storage.tool_use_log import ToolUseLogger
            self.tool_logger = ToolUseLogger()
        except ImportError:
            self.logger.warning("ToolUseLogger not available, tool logging disabled")
            self.tool_logger = None
    
    def _register_builtin_tools(self):
        """Register built-in system tools."""
        # Shell command execution
        self.register_tool('run_shell_command', self._run_shell_command, 
                          description="Execute any shell command with full system access")
        
        # Package installation
        self.register_tool('install_package', self._install_package,
                          description="Install packages using pip")
        
        # File operations
        self.register_tool('read_file', self._read_file,
                          description="Read file contents")
        self.register_tool('write_file', self._write_file,
                          description="Write content to file")
        self.register_tool('delete_file', self._delete_file,
                          description="Delete a file")
        self.register_tool('list_directory', self._list_directory,
                          description="List directory contents")
        
        # Process management
        self.register_tool('kill_process', self._kill_process,
                          description="Kill a process by PID")
        self.register_tool('list_processes', self._list_processes,
                          description="List running processes")
        
        # System operations
        if self.unrestricted_mode:
            self.register_tool('reboot_system', self._reboot_system,
                              description="Reboot the system")
            self.register_tool('update_system', self._update_system,
                              description="Update system packages")
        
        self.logger.info(f"[ToolRegistry] Registered {len(self.tools)} built-in tools")
    
    def register_tool(self, name: str, func: Callable, description: str = ""):
        """
        Register a tool function.
        
        Args:
            name: Tool name
            func: Tool function
            description: Tool description
        """
        self.tools[name] = {
            'function': func,
            'description': description,
            'registered_at': datetime.now().isoformat()
        }
        self.logger.info(f"[ToolRegistry] Registered tool: {name}")
    
    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            name: Tool name
            
        Returns:
            True if tool was removed, False if not found
        """
        if name in self.tools:
            del self.tools[name]
            self.logger.info(f"[ToolRegistry] Unregistered tool: {name}")
            return True
        return False
    
    def get_tool_list(self) -> List[Dict[str, Any]]:
        """Get list of all registered tools."""
        return [
            {
                'name': name,
                'description': tool_info['description'],
                'registered_at': tool_info['registered_at']
            }
            for name, tool_info in self.tools.items()
        ]
    
    async def execute_tool(self, tool_name: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute a registered tool.
        
        Args:
            tool_name: Name of the tool to execute
            *args: Tool arguments
            **kwargs: Tool keyword arguments
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            return {
                'success': False,
                'error': f"Tool '{tool_name}' not found",
                'available_tools': list(self.tools.keys())
            }
        
        tool_info = self.tools[tool_name]
        start_time = time.time()
        
        try:
            # Execute the tool function
            if asyncio.iscoroutinefunction(tool_info['function']):
                result = await tool_info['function'](*args, **kwargs)
            else:
                result = tool_info['function'](*args, **kwargs)
            
            execution_result = {
                'success': True,
                'result': result,
                'tool_name': tool_name,
                'duration': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log tool usage
            if self.tool_logger:
                self.tool_logger.log_tool_usage(
                    tool_name=tool_name,
                    executor="system",
                    tool_args=str(args) + str(kwargs),
                    success=True,
                    result=result,
                    duration=execution_result['duration']
                )
            
            return execution_result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'tool_name': tool_name,
                'duration': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log tool usage error
            if self.tool_logger:
                self.tool_logger.log_tool_usage(
                    tool_name=tool_name,
                    executor="system",
                    tool_args=str(args) + str(kwargs),
                    success=False,
                    error=str(e),
                    duration=error_result['duration']
                )
            
            self.logger.error(f"[ToolRegistry] Error executing tool {tool_name}: {e}")
            return error_result
    
    # Built-in tool implementations
    
    async def _run_shell_command(self, command: str, cwd: str = None, 
                                env: Dict[str, str] = None, 
                                timeout: int = 30) -> Dict[str, Any]:
        """Execute shell command with streaming support."""
        if not self.enable_shell_commands:
            return {'error': 'Shell commands disabled in configuration'}
        
        self.logger.info(f"[ToolRegistry] Executing shell command: {command}")
        
        try:
            # Set up environment
            cmd_env = os.environ.copy()
            if env:
                cmd_env.update(env)
            
            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=cmd_env
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    'error': f'Command timed out after {timeout} seconds',
                    'exit_code': -1
                }
            
            return {
                'stdout': stdout.decode('utf-8', errors='replace'),
                'stderr': stderr.decode('utf-8', errors='replace'),
                'exit_code': process.returncode,
                'success': process.returncode == 0
            }
            
        except Exception as e:
            return {'error': f'Shell execution error: {str(e)}'}
    
    async def _install_package(self, package: str, upgrade: bool = False, 
                              force: bool = False) -> Dict[str, Any]:
        """Install Python package using pip."""
        cmd = f"pip install {package}"
        if upgrade:
            cmd += " --upgrade"
        if force:
            cmd += " --force-reinstall"
        
        return await self._run_shell_command(cmd)
    
    def _read_file(self, file_path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
        """Read file contents."""
        if not self.enable_file_operations:
            return {'error': 'File operations disabled in configuration'}
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            return {
                'content': content,
                'size': len(content),
                'path': file_path
            }
        except Exception as e:
            return {'error': f'File read error: {str(e)}'}
    
    def _write_file(self, file_path: str, content: str, 
                   encoding: str = 'utf-8', mode: str = 'w') -> Dict[str, Any]:
        """Write content to file."""
        if not self.enable_file_operations:
            return {'error': 'File operations disabled in configuration'}
        
        try:
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)
            return {
                'success': True,
                'path': file_path,
                'size': len(content)
            }
        except Exception as e:
            return {'error': f'File write error: {str(e)}'}
    
    def _delete_file(self, file_path: str) -> Dict[str, Any]:
        """Delete a file."""
        if not self.enable_file_operations:
            return {'error': 'File operations disabled in configuration'}
        
        try:
            os.remove(file_path)
            return {'success': True, 'path': file_path}
        except Exception as e:
            return {'error': f'File delete error: {str(e)}'}
    
    def _list_directory(self, dir_path: str = ".") -> Dict[str, Any]:
        """List directory contents."""
        try:
            items = []
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                items.append({
                    'name': item,
                    'is_file': os.path.isfile(item_path),
                    'is_dir': os.path.isdir(item_path),
                    'size': os.path.getsize(item_path) if os.path.isfile(item_path) else None
                })
            return {'items': items, 'path': dir_path}
        except Exception as e:
            return {'error': f'Directory list error: {str(e)}'}
    
    async def _kill_process(self, pid: int) -> Dict[str, Any]:
        """Kill a process by PID."""
        if not self.unrestricted_mode:
            return {'error': 'Process killing disabled in restricted mode'}
        
        cmd = f"kill -9 {pid}"
        return await self._run_shell_command(cmd)
    
    async def _list_processes(self) -> Dict[str, Any]:
        """List running processes."""
        cmd = "ps aux"
        return await self._run_shell_command(cmd)
    
    async def _reboot_system(self) -> Dict[str, Any]:
        """Reboot the system."""
        if not self.unrestricted_mode:
            return {'error': 'System reboot disabled in restricted mode'}
        
        self.logger.warning("[ToolRegistry] System reboot requested!")
        cmd = "sudo reboot"
        return await self._run_shell_command(cmd)
    
    async def _update_system(self) -> Dict[str, Any]:
        """Update system packages."""
        if not self.unrestricted_mode:
            return {'error': 'System update disabled in restricted mode'}
        
        cmd = "sudo apt update && sudo apt upgrade -y"
        return await self._run_shell_command(cmd, timeout=300)  # 5 minute timeout
    
    def get_background_processes(self) -> Dict[str, Any]:
        """Get list of background processes."""
        return self.background_processes.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get tool registry status."""
        return {
            'enabled': True,
            'unrestricted_mode': self.unrestricted_mode,
            'shell_commands_enabled': self.enable_shell_commands,
            'file_operations_enabled': self.enable_file_operations,
            'background_processes_enabled': self.enable_background_processes,
            'total_tools': len(self.tools),
            'tool_names': list(self.tools.keys()),
            'background_processes': len(self.background_processes),
            'config': self.config
        }


# Global tool registry instance
_tool_registry = None

def get_tool_registry() -> ToolRegistry:
    """Get or create global tool registry instance."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry 