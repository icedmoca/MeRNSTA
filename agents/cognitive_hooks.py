#!/usr/bin/env python3
"""
Cognitive Hooks - Integration points for CognitiveArbiter routing

Provides hooks that can be inserted into decision paths throughout the system
to enable dynamic arbitration between cognitive modes.
"""

import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime

# Global arbiter instance (lazy loaded)
_arbiter_instance = None

def get_cognitive_arbiter():
    """Lazy load the cognitive arbiter instance"""
    global _arbiter_instance
    if _arbiter_instance is None:
        try:
            from agents.registry import get_agent_registry
            registry = get_agent_registry()
            _arbiter_instance = registry.get_agent('cognitive_arbiter')
        except ImportError:
            logging.debug("CognitiveArbiter not available - using fallback")
            return None
    return _arbiter_instance


def cognitive_arbitration_hook(message: str, context: Dict[str, Any] = None, 
                             default_handler: Optional[Callable] = None) -> Any:
    """
    Main cognitive arbitration hook.
    
    Intercepts decision points and routes to the appropriate cognitive agent
    based on current cognitive state, dissonance, confidence, etc.
    
    Args:
        message: The input message/request to process
        context: Additional context for decision making
        default_handler: Fallback handler if arbitration fails
    
    Returns:
        Response from the chosen cognitive agent
    """
    if not context:
        context = {}
    
    # Add hook metadata
    context['hook_type'] = 'cognitive_arbitration'
    context['hook_timestamp'] = datetime.now().isoformat()
    
    try:
        arbiter = get_cognitive_arbiter()
        
        if arbiter:
            # Route through cognitive arbiter
            logging.debug(f"[CognitiveHook] Routing through arbiter: {message[:50]}...")
            return arbiter.respond(message, context)
        else:
            # Fallback to default handler
            if default_handler:
                logging.debug(f"[CognitiveHook] Using default handler: {message[:50]}...")
                return default_handler(message, context)
            else:
                logging.warning(f"[CognitiveHook] No arbiter or default handler available")
                return "I need to process this request, but cognitive routing is not available."
    
    except Exception as e:
        logging.error(f"[CognitiveHook] Error in arbitration: {e}")
        
        # Try default handler as fallback
        if default_handler:
            try:
                return default_handler(message, context)
            except Exception as fallback_error:
                logging.error(f"[CognitiveHook] Default handler also failed: {fallback_error}")
        
        return f"Error processing request: {e}"


def agent_loop_hook(message: str, context: Dict[str, Any] = None) -> Any:
    """
    Hook for main agent processing loop.
    
    This should be called in the main conversation/agent loop to enable
    cognitive arbitration for all user interactions.
    """
    context = context or {}
    context['hook_source'] = 'agent_loop'
    
    def default_simple_response(msg, ctx):
        """Simple fallback response"""
        return f"Processing: {msg}"
    
    return cognitive_arbitration_hook(message, context, default_simple_response)


def planner_hook(message: str, context: Dict[str, Any] = None) -> Any:
    """
    Hook for planning system decisions.
    
    Determines whether to use fast reflex, structured planning, or deep meta-cognition
    for planning tasks.
    """
    context = context or {}
    context['hook_source'] = 'planner'
    context['requires_planning'] = True
    
    def default_planner_response(msg, ctx):
        """Fallback to simple planning response"""
        return f"Planning approach for: {msg}"
    
    return cognitive_arbitration_hook(message, context, default_planner_response)


def meta_decision_hook(message: str, context: Dict[str, Any] = None) -> Any:
    """
    Hook for meta-level decisions.
    
    Used for high-level cognitive decisions about system behavior,
    self-modification, learning, etc.
    """
    context = context or {}
    context['hook_source'] = 'meta_decision'
    context['meta_level'] = True
    context['high_stakes'] = True
    
    def default_meta_response(msg, ctx):
        """Fallback to meta-level response"""
        return f"Meta-level analysis of: {msg}"
    
    return cognitive_arbitration_hook(message, context, default_meta_response)


def response_generation_hook(message: str, context: Dict[str, Any] = None) -> Any:
    """
    Hook for response generation.
    
    Determines the appropriate cognitive mode for generating responses
    to user queries.
    """
    context = context or {}
    context['hook_source'] = 'response_generation'
    context['user_facing'] = True
    
    def default_response_generation(msg, ctx):
        """Fallback response generation"""
        return f"Response to: {msg}"
    
    return cognitive_arbitration_hook(message, context, default_response_generation)


def problem_solving_hook(message: str, context: Dict[str, Any] = None) -> Any:
    """
    Hook for problem-solving tasks.
    
    Routes complex problem-solving to appropriate cognitive agents
    based on problem complexity, time constraints, etc.
    """
    context = context or {}
    context['hook_source'] = 'problem_solving'
    context['requires_analysis'] = True
    
    def default_problem_solving(msg, ctx):
        """Fallback problem solving"""
        return f"Analyzing problem: {msg}"
    
    return cognitive_arbitration_hook(message, context, default_problem_solving)


def error_handling_hook(error: Exception, message: str, context: Dict[str, Any] = None) -> Any:
    """
    Hook for error handling decisions.
    
    Determines how to handle errors - quick recovery vs. deep analysis
    of the error condition.
    """
    context = context or {}
    context['hook_source'] = 'error_handling'
    context['error_type'] = type(error).__name__
    context['error_message'] = str(error)
    context['time_pressure'] = 5.0  # Quick error recovery preferred
    
    error_analysis_message = f"Error handling required: {error} for request: {message}"
    
    def default_error_handling(msg, ctx):
        """Fallback error handling"""
        return f"Error occurred: {ctx.get('error_type', 'Unknown')} - using fallback handling"
    
    return cognitive_arbitration_hook(error_analysis_message, context, default_error_handling)


def learning_hook(experience: str, context: Dict[str, Any] = None) -> Any:
    """
    Hook for learning and adaptation decisions.
    
    Determines how to process new experiences and learning opportunities.
    """
    context = context or {}
    context['hook_source'] = 'learning'
    context['learning_opportunity'] = True
    
    learning_message = f"Learning from experience: {experience}"
    
    def default_learning(msg, ctx):
        """Fallback learning"""
        return f"Learning processed: {experience}"
    
    return cognitive_arbitration_hook(learning_message, context, default_learning)


def register_arbitration_hooks():
    """
    Register arbitration hooks with various system components.
    
    This function should be called during system initialization to set up
    cognitive arbitration throughout the system.
    """
    try:
        # This could register hooks with various subsystems
        # For now, we just ensure the arbiter is available
        arbiter = get_cognitive_arbiter()
        if arbiter:
            logging.info("[CognitiveHooks] Cognitive arbitration hooks registered successfully")
            return True
        else:
            logging.warning("[CognitiveHooks] CognitiveArbiter not available - hooks not registered")
            return False
    except Exception as e:
        logging.error(f"[CognitiveHooks] Error registering hooks: {e}")
        return False


def arbitration_enabled() -> bool:
    """Check if cognitive arbitration is available and enabled"""
    arbiter = get_cognitive_arbiter()
    return arbiter is not None


# Convenience decorator for adding arbitration to functions
def with_cognitive_arbitration(hook_type: str = 'general'):
    """
    Decorator to add cognitive arbitration to any function.
    
    Usage:
        @with_cognitive_arbitration('planning')
        def my_planning_function(message, context=None):
            return original_logic(message, context)
    """
    def decorator(func):
        def wrapper(message, context=None, *args, **kwargs):
            if arbitration_enabled():
                context = context or {}
                context['hook_type'] = hook_type
                context['original_function'] = func.__name__
                
                return cognitive_arbitration_hook(
                    message, 
                    context, 
                    lambda msg, ctx: func(msg, ctx, *args, **kwargs)
                )
            else:
                return func(message, context, *args, **kwargs)
        return wrapper
    return decorator