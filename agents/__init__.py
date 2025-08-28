"""
MeRNSTA Multi-Agent Cognitive System

This package contains all the cognitive agents and the registry system.
"""

from .registry import get_agent_registry, AgentRegistry, AGENT_REGISTRY
from .planner import PlannerAgent
from .critic import CriticAgent
from .debater import DebaterAgent
from .reflector import ReflectorAgent

__all__ = [
    'get_agent_registry',
    'AgentRegistry', 
    'AGENT_REGISTRY',
    'PlannerAgent',
    'CriticAgent', 
    'DebaterAgent',
    'ReflectorAgent'
] 