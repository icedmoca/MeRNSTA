#!/usr/bin/env python3
"""
Base Agent Class for MeRNSTA Multi-Agent Cognitive System

Provides shared functionality and interfaces for all cognitive agents.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from config.settings import get_config
from .agent_contract import load_or_create_contract, AgentContract

class BaseAgent(ABC):
    """
    Abstract base class for all MeRNSTA cognitive agents.
    
    Provides:
    - LLM integration via existing infrastructure
    - Configuration loading
    - Standardized response interface
    - Memory access capabilities
    """
    
    def __init__(self, name: str):
        self.name = name
        self.config = get_config().get('multi_agent', {})
        self.agent_config = self.config.get('agent_configs', {}).get(name, {})
        self.enabled = self.config.get('enabled', True)
        
        # Initialize LLM fallback for text generation
        self._llm_fallback = None
        self._symbolic_engine = None
        self._memory_system = None
        self._personality_engine = None
        
        # Load agent contract for declarative role specification
        self.contract: Optional[AgentContract] = None
        self._load_contract()
        
        # Lifecycle tracking fields
        self.last_promotion: Optional[datetime] = None
        self.last_mutation: Optional[datetime] = None
        self.lifecycle_history: List[Dict[str, Any]] = []
        
        # Performance tracking for lifecycle management
        self._execution_count: int = 0
        self._success_count: int = 0
        self._failure_count: int = 0
        self._confidence_history: List[Dict[str, float]] = []
        
        logging.info(f"[{self.name}Agent] Initialized with config: {self.agent_config}")
    
    @property
    def llm_fallback(self):
        """Lazy-load LLM fallback agent."""
        if self._llm_fallback is None:
            try:
                from tools.llm_fallback import LLMFallbackAgent
                self._llm_fallback = LLMFallbackAgent()
            except ImportError as e:
                logging.error(f"[{self.name}Agent] Could not load LLM fallback: {e}")
                self._llm_fallback = None
        return self._llm_fallback
    
    @property
    def symbolic_engine(self):
        """Lazy-load symbolic reasoning engine."""
        if self._symbolic_engine is None:
            try:
                from tools.symbolic_engine import SymbolicEngine
                self._symbolic_engine = SymbolicEngine()
            except ImportError as e:
                logging.error(f"[{self.name}Agent] Could not load symbolic engine: {e}")
                self._symbolic_engine = None
        return self._symbolic_engine
    
    @property
    def memory_system(self):
        """Lazy-load memory system for context."""
        if self._memory_system is None:
            try:
                from storage.phase2_cognitive_system import Phase2AutonomousCognitiveSystem
                self._memory_system = Phase2AutonomousCognitiveSystem()
            except ImportError as e:
                logging.error(f"[{self.name}Agent] Could not load memory system: {e}")
                self._memory_system = None
        return self._memory_system
    
    @property
    def personality(self):
        """Lazy-load personality engine for dynamic personality evolution."""
        if self._personality_engine is None:
            try:
                from agents.personality_engine import get_personality_engine
                self._personality_engine = get_personality_engine()
            except ImportError as e:
                logging.error(f"[{self.name}Agent] Could not load personality engine: {e}")
                self._personality_engine = None
        return self._personality_engine
    
    def generate_llm_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """
        Generate a response using the LLM fallback system.
        
        Args:
            prompt: The prompt to send to the LLM
            context: Additional context for the response
            
        Returns:
            Generated response text
        """
        if not self.llm_fallback:
            return f"[{self.name}Agent] LLM not available"
            
        try:
            result = self.llm_fallback.generate_response(prompt, context)
            if result.get('success'):
                return result.get('response', f"[{self.name}Agent] No response generated")
            else:
                return f"[{self.name}Agent] Error: {result.get('error', 'Unknown error')}"
        except Exception as e:
            logging.error(f"[{self.name}Agent] LLM generation error: {e}")
            return f"[{self.name}Agent] Error generating response"
    
    def get_memory_context(self, query: str, max_facts: int = 5) -> List[str]:
        """
        Get relevant memory context for a query.
        
        Args:
            query: The query to search memory for
            max_facts: Maximum number of facts to return
            
        Returns:
            List of relevant memory facts as strings
        """
        if not self.memory_system:
            return []
            
        try:
            # Get facts from memory system
            facts = self.memory_system.get_facts(user_profile_id="agent_context")
            
            # Simple relevance scoring (could be enhanced)
            relevant_facts = []
            query_words = set(query.lower().split())
            
            for fact in facts[:max_facts * 2]:  # Get more to filter
                fact_text = f"{fact.subject} {fact.predicate} {fact.object}".lower()
                fact_words = set(fact_text.split())
                
                # Calculate simple overlap score
                overlap = len(query_words.intersection(fact_words))
                if overlap > 0:
                    relevant_facts.append((fact_text, overlap))
            
            # Sort by relevance and return top results
            relevant_facts.sort(key=lambda x: x[1], reverse=True)
            return [fact[0] for fact in relevant_facts[:max_facts]]
            
        except Exception as e:
            logging.error(f"[{self.name}Agent] Memory context error: {e}")
            return []
    
    def style_response_with_personality(self, response: str, context: Dict[str, Any] = None) -> str:
        """
        Apply personality styling to a response.
        
        Args:
            response: Base response text
            context: Additional context for personality adjustments
            
        Returns:
            Personality-styled response
        """
        if not self.personality:
            return response
            
        try:
            # Check for emotional dissonance and evolve if needed
            if context:
                self.personality.check_for_emotional_dissonance(context)
            
            # Apply personality styling
            styled_response = self.personality.style_response(response, context)
            return styled_response
            
        except Exception as e:
            logging.error(f"[{self.name}Agent] Error styling response with personality: {e}")
            return response
    
    def build_agent_prompt(self, message: str, memory_context: List[str] = None) -> str:
        """
        Build a specialized prompt for this agent type.
        
        Args:
            message: The user message
            memory_context: Relevant memory facts
            
        Returns:
            Formatted prompt for the agent
        """
        prompt_parts = [
            f"You are the {self.name}Agent, a specialized cognitive agent in the MeRNSTA system.",
            self.get_agent_instructions(),
            "",
            f"User message: {message}"
        ]
        
        if memory_context:
            prompt_parts.insert(-1, f"Relevant context: {'; '.join(memory_context)}")
            prompt_parts.insert(-1, "")
            
        base = "\n".join(prompt_parts)
        # Enforce token budget for base prompts
        try:
            from config.settings import get_token_budget
            from utils.prompt_budget import truncate_by_tokens
            budget = max(16, int(get_token_budget()))
            return truncate_by_tokens(base, budget)
        except Exception:
            return base
    
    @abstractmethod
    def get_agent_instructions(self) -> str:
        """
        Get the specialized instructions for this agent type.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """
        Generate a response as this agent type.
        Must be implemented by subclasses.
        
        Args:
            message: The input message
            context: Additional context
            
        Returns:
            Agent response
        """
        pass
    
    def _load_contract(self) -> None:
        """Load the agent contract for declarative role specification."""
        try:
            # Clean up agent name for contract creation
            clean_name = self._get_clean_agent_name()
            self.contract = load_or_create_contract(clean_name)
            
            # Check for contract version drift or staleness
            if self.contract:
                from datetime import datetime
                contract_age_days = (datetime.now() - self.contract.last_updated).days
                if contract_age_days > 30:
                    logging.warning(f"[{self.name}Agent] Contract is {contract_age_days} days old - consider updating")
                
                # Log contract loading
                logging.info(f"[{self.name}Agent] Loaded contract v{self.contract.version} - {self.contract.purpose}")
            else:
                logging.error(f"[{self.name}Agent] Failed to load contract")
                
        except Exception as e:
            logging.error(f"[{self.name}Agent] Error loading contract: {e}")
            self.contract = None
    
    def _get_clean_agent_name(self) -> str:
        """Get a clean agent name suitable for contract creation."""
        # Convert to string if it's an object
        if hasattr(self.name, '__str__'):
            clean_name = str(self.name)
        else:
            clean_name = self.name
        
        # Remove common prefixes
        prefixes_to_remove = [
            '<storage.phase2_cognitive_system.Phase2AutonomousCognitiveSystem object at ',
            '<',
            '>',
            ' object at '
        ]
        
        for prefix in prefixes_to_remove:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
            if clean_name.endswith(prefix):
                clean_name = clean_name[:-len(prefix)]
        
        # Remove memory addresses (hex strings)
        import re
        clean_name = re.sub(r'0x[0-9a-f]+', '', clean_name)
        
        # Clean up any remaining artifacts
        clean_name = clean_name.strip()
        
        # If we end up with an empty name, use a default
        if not clean_name:
            clean_name = 'phase2_cognitive_system'
        
        return clean_name
    
    def score_task_alignment(self, task: str, context: Dict[str, Any] = None) -> float:
        """
        Score how well this agent aligns with a given task.
        
        Args:
            task: Task description
            context: Additional task context
            
        Returns:
            Alignment score between 0.0 and 1.0
        """
        if not self.contract:
            logging.warning(f"[{self.name}Agent] No contract available for task alignment scoring")
            return 0.5  # Default neutral score
        
        try:
            # Create task object with context if available
            if context:
                task_obj = {
                    'description': task,
                    'type': context.get('type'),
                    'urgency': context.get('urgency', 0.5),
                    'complexity': context.get('complexity'),
                    'keywords': context.get('keywords')
                }
                return self.contract.score_alignment(task_obj)
            else:
                return self.contract.score_alignment(task)
                
        except Exception as e:
            logging.error(f"[{self.name}Agent] Error scoring task alignment: {e}")
            return 0.0
    
    def update_contract_from_performance(self, task_result: Dict[str, Any]) -> None:
        """
        Update contract based on task performance feedback.
        
        Args:
            task_result: Dictionary containing task outcome and performance metrics
        """
        if not self.contract:
            logging.warning(f"[{self.name}Agent] No contract available for performance update")
            return
        
        try:
            self.contract.update_from_performance_feedback(task_result)
            
            # Save updated contract
            self.contract.save_to_file(f"output/contracts/{self.name}.json")
            
            logging.info(f"[{self.name}Agent] Updated contract based on performance feedback")
            
        except Exception as e:
            logging.error(f"[{self.name}Agent] Error updating contract from performance: {e}")
    
    def get_contract_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's contract."""
        if not self.contract:
            return {"error": "No contract available"}
        
        return self.contract.get_summary()
    
    def reload_contract(self) -> bool:
        """
        Reload the agent contract from file.
        
        Returns:
            True if contract was successfully reloaded, False otherwise
        """
        try:
            self._load_contract()
            return self.contract is not None
        except Exception as e:
            logging.error(f"[{self.name}Agent] Error reloading contract: {e}")
            return False

    def get_capabilities(self) -> Dict[str, Any]:
        """Return information about this agent's capabilities."""
        base_capabilities = {
            "name": self.name,
            "enabled": self.enabled,
            "has_llm": self.llm_fallback is not None,
            "has_symbolic": self.symbolic_engine is not None,
            "has_memory": self.memory_system is not None,
            "has_personality": self.personality is not None,
            "config": self.agent_config
        }
        
        # Add contract information if available
        if self.contract:
            base_capabilities.update({
                "contract_loaded": True,
                "contract_version": self.contract.version,
                "declared_purpose": self.contract.purpose,
                "declared_capabilities": self.contract.capabilities,
                "confidence_vector": self.contract.confidence_vector,
                "last_contract_update": self.contract.last_updated.isoformat()
            })
        else:
            base_capabilities["contract_loaded"] = False
        
        return base_capabilities
    
    def get_lifecycle_metrics(self) -> Dict[str, Any]:
        """
        Get lifecycle metrics for this agent used by the lifecycle manager.
        
        Returns:
            Dict containing agent performance and status metrics
        """
        try:
            # Calculate success rate
            total_executions = self._execution_count
            success_rate = self._success_count / total_executions if total_executions > 0 else 0.0
            
            # Get current confidence vector from contract
            confidence_vector = {}
            if self.contract and self.contract.confidence_vector:
                confidence_vector = self.contract.confidence_vector.copy()
            
            # Calculate current performance (recent success rate)
            recent_window = 10
            if len(self.lifecycle_history) >= recent_window:
                recent_successes = sum(1 for entry in self.lifecycle_history[-recent_window:] 
                                     if entry.get('success', False))
                current_performance = recent_successes / recent_window
            else:
                current_performance = success_rate
            
            # Calculate confidence variance (stability metric)
            confidence_variance = 0.0
            if len(self._confidence_history) > 1:
                try:
                    import numpy as np
                    # Calculate variance across confidence values for common keys
                    if self._confidence_history:
                        all_keys = set()
                        for conf in self._confidence_history:
                            all_keys.update(conf.keys())
                        
                        variances = []
                        for key in all_keys:
                            values = [conf.get(key, 0.0) for conf in self._confidence_history]
                            if len(values) > 1:
                                variances.append(np.var(values))
                        
                        confidence_variance = np.mean(variances) if variances else 0.0
                except ImportError:
                    # Fallback if numpy not available
                    confidence_variance = 0.1
            
            return {
                'execution_count': total_executions,
                'success_count': self._success_count,
                'failure_count': self._failure_count,
                'success_rate': success_rate,
                'current_performance': current_performance,
                'confidence_vector': confidence_vector,
                'confidence_variance': confidence_variance,
                'last_promotion': self.last_promotion.isoformat() if self.last_promotion else None,
                'last_mutation': self.last_mutation.isoformat() if self.last_mutation else None,
                'lifecycle_history_length': len(self.lifecycle_history),
                'agent_name': self.name,
                'contract_version': self.contract.version if self.contract else None
            }
            
        except Exception as e:
            logging.error(f"[{self.name}Agent] Error getting lifecycle metrics: {e}")
            return {
                'execution_count': 0,
                'success_rate': 0.0,
                'current_performance': 0.0,
                'confidence_vector': {},
                'confidence_variance': 0.0,
                'agent_name': self.name,
                'error': str(e)
            }
    
    def record_execution(self, success: bool, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Record an execution for lifecycle tracking.
        
        Args:
            success: Whether the execution was successful
            context: Optional context about the execution
        """
        try:
            self._execution_count += 1
            
            if success:
                self._success_count += 1
            else:
                self._failure_count += 1
            
            # Record in lifecycle history
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'success': success,
                'execution_id': self._execution_count
            }
            
            if context:
                history_entry['context'] = context
            
            self.lifecycle_history.append(history_entry)
            
            # Keep only recent history (last 100 entries)
            if len(self.lifecycle_history) > 100:
                self.lifecycle_history = self.lifecycle_history[-100:]
            
            # Update confidence history if contract exists
            if self.contract and self.contract.confidence_vector:
                self._confidence_history.append(self.contract.confidence_vector.copy())
                
                # Keep only recent confidence history
                if len(self._confidence_history) > 50:
                    self._confidence_history = self._confidence_history[-50:]
            
            logging.debug(f"[{self.name}Agent] Recorded execution: success={success}, total={self._execution_count}")
            
        except Exception as e:
            logging.error(f"[{self.name}Agent] Error recording execution: {e}")
    
    def update_lifecycle_event(self, event_type: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Update lifecycle tracking with significant events.
        
        Args:
            event_type: Type of lifecycle event ('promotion', 'mutation', 'retirement', etc.)
            details: Optional details about the event
        """
        try:
            timestamp = datetime.now()
            
            if event_type == 'promotion':
                self.last_promotion = timestamp
            elif event_type == 'mutation':
                self.last_mutation = timestamp
            
            # Record in lifecycle history
            history_entry = {
                'timestamp': timestamp.isoformat(),
                'event_type': event_type,
                'agent_name': self.name
            }
            
            if details:
                history_entry['details'] = details
            
            self.lifecycle_history.append(history_entry)
            
            logging.info(f"[{self.name}Agent] Lifecycle event recorded: {event_type}")
            
        except Exception as e:
            logging.error(f"[{self.name}Agent] Error updating lifecycle event: {e}")
    
    def __str__(self):
        return f"{self.name}Agent"
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class Goal:
    """Simple goal class for backward compatibility."""
    
    def __init__(self, description: str, priority: float = 0.5, deadline: Optional[datetime] = None):
        self.description = description
        self.priority = priority
        self.deadline = deadline
        self.created_at = datetime.now()
        self.completed = False
        
    def __str__(self):
        return f"Goal('{self.description}', priority={self.priority})"
        
    def to_dict(self):
        return {
            'description': self.description,
            'priority': self.priority,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'created_at': self.created_at.isoformat(),
            'completed': self.completed
        }


# Backward compatibility aliases
Agent = BaseAgent 