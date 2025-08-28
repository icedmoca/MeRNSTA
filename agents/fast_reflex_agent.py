#!/usr/bin/env python3
"""
Fast Reflex Agent for MeRNSTA Phase 31

A lightweight, real-time "reflex" layer for instantaneous reactions without full deliberation.
Borrowed from HRM's fast forward-pass module design.

Key Features:
- Operates on shallow heuristics for rapid response
- Triggers on cognitive load, timeouts, or low-effort tasks
- Shallow memory scan and pre-trained heuristic maps
- Can defer to deep planning when needed
- Recent actions as priming
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum

from config.settings import get_config
from .base import BaseAgent


class ReflexTriggerType(Enum):
    """Types of triggers that activate the reflex agent."""
    COGNITIVE_LOAD_HIGH = "cognitive_load_high"
    TIMEOUT_THRESHOLD = "timeout_threshold" 
    LOW_EFFORT_TASK = "low_effort_task"
    MANUAL_ACTIVATION = "manual_activation"


class ReflexResponse(Enum):
    """Types of reflex responses."""
    IMMEDIATE_ACTION = "immediate_action"
    CACHED_RESPONSE = "cached_response"
    HEURISTIC_MATCH = "heuristic_match"
    DEFER_TO_DEEP = "defer_to_deep"


@dataclass
class HeuristicPattern:
    """Pre-trained heuristic pattern for rapid matching."""
    pattern_id: str
    trigger_keywords: List[str]
    response_template: str
    confidence_score: float
    usage_count: int = 0
    success_rate: float = 1.0
    last_used: Optional[datetime] = None
    context_requirements: List[str] = field(default_factory=list)


@dataclass
class ReflexAction:
    """Action taken by the reflex agent."""
    action_type: ReflexResponse
    response_text: str
    confidence: float
    trigger_type: ReflexTriggerType
    processing_time_ms: float
    heuristic_used: Optional[str] = None
    deferred_to_deep: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CognitiveLoadMetrics:
    """Metrics for assessing cognitive load."""
    active_agents: int
    pending_tasks: int
    memory_usage_percent: float
    response_latency_ms: float
    error_rate: float
    complexity_score: float
    timestamp: datetime = field(default_factory=datetime.now)


class FastReflexAgent(BaseAgent):
    """
    Lightweight reflex agent for instantaneous reactions.
    
    Provides rapid responses using shallow heuristics, cached patterns,
    and recent action priming without deep deliberation.
    """
    
    def __init__(self):
        super().__init__("fast_reflex")
        
        # Load configuration
        self.config = get_config()
        self.reflex_config = self.config.get('fast_reflex', {})
        
        # Reflex mode state
        self.reflex_mode_enabled = self.reflex_config.get('enabled', False)
        self.auto_trigger_enabled = self.reflex_config.get('auto_trigger', True)
        
        # Cognitive load thresholds (no hardcoding per user's memory)
        self.cognitive_load_threshold = self.reflex_config.get('cognitive_load_threshold', 0.8)
        self.timeout_threshold_ms = self.reflex_config.get('timeout_threshold_ms', 2000)
        self.low_effort_keywords = self.reflex_config.get('low_effort_keywords', [
            'yes', 'no', 'ok', 'thanks', 'hello', 'help', 'status', 'list'
        ])
        
        # Response timing thresholds
        self.max_reflex_time_ms = self.reflex_config.get('max_reflex_time_ms', 100)
        self.defer_threshold = self.reflex_config.get('defer_threshold', 0.3)
        
        # Memory and heuristics
        self.heuristic_patterns: Dict[str, HeuristicPattern] = {}
        self.response_cache: Dict[str, ReflexAction] = {}
        self.recent_actions = deque(maxlen=50)
        self.shallow_memory_limit = self.reflex_config.get('shallow_memory_limit', 20)
        
        # Performance tracking
        self.response_history = deque(maxlen=100)
        self.cognitive_load_history = deque(maxlen=20)
        self.success_rate_window = deque(maxlen=50)
        
        # Integration components
        self._autonomous_planner = None
        self._memory_system = None
        self._meta_self_agent = None
        
        # Load pre-trained heuristics
        self._load_heuristic_patterns()
        
        logging.info(f"[{self.name}] Initialized with reflex_mode={'enabled' if self.reflex_mode_enabled else 'disabled'}")
    
    def get_agent_instructions(self) -> str:
        """Get instructions for the FastReflexAgent."""
        return """You are the FastReflexAgent, responsible for instantaneous reactions using shallow heuristics.

Your primary functions:
1. Provide rapid responses to low-effort queries without deep deliberation
2. Use cached patterns and heuristic matching for immediate reactions
3. Monitor cognitive load and trigger when system is overwhelmed
4. Defer to deep planning when complexity exceeds reflex capabilities
5. Maintain response cache and heuristic pattern database

Focus on speed, simplicity, and knowing when to defer to deeper systems."""
    
    def toggle_reflex_mode(self, enabled: bool) -> Dict[str, Any]:
        """Toggle reflex mode on/off."""
        old_state = self.reflex_mode_enabled
        self.reflex_mode_enabled = enabled
        
        status_msg = f"Reflex mode {'enabled' if enabled else 'disabled'}"
        logging.info(f"[{self.name}] {status_msg}")
        
        return {
            "status": "success",
            "message": status_msg,
            "previous_state": old_state,
            "current_state": enabled,
            "timestamp": datetime.now().isoformat()
        }
    
    def assess_cognitive_load(self) -> CognitiveLoadMetrics:
        """Assess current cognitive load of the system."""
        try:
            # Get agent registry for active agent count
            from agents.registry import AgentRegistry
            registry = AgentRegistry()
            active_agents = len(registry.get_all_agents())
            
            # Get task queue length (if available)
            pending_tasks = 0
            try:
                if hasattr(self, '_task_selector') and self._task_selector:
                    pending_tasks = len(getattr(self._task_selector, 'task_queue', []))
            except:
                pass
            
            # Estimate memory usage (simplified)
            memory_usage = len(self.recent_actions) / 50.0 * 100
            
            # Calculate average response latency
            recent_latencies = [action.processing_time_ms for action in list(self.recent_actions)[-10:]]
            avg_latency = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0
            
            # Calculate error rate
            recent_successes = [1 for rate in list(self.success_rate_window)[-10:] if rate > 0.7]
            error_rate = 1.0 - (len(recent_successes) / min(10, len(self.success_rate_window))) if self.success_rate_window else 0
            
            # Complexity score based on recent task types
            complexity_indicators = ['plan', 'analyze', 'debug', 'implement', 'refactor']
            recent_complexities = []
            for action in list(self.recent_actions)[-5:]:
                complexity = sum(1 for indicator in complexity_indicators 
                               if indicator in action.response_text.lower()) / len(complexity_indicators)
                recent_complexities.append(complexity)
            
            complexity_score = sum(recent_complexities) / len(recent_complexities) if recent_complexities else 0
            
            metrics = CognitiveLoadMetrics(
                active_agents=active_agents,
                pending_tasks=pending_tasks,
                memory_usage_percent=memory_usage,
                response_latency_ms=avg_latency,
                error_rate=error_rate,
                complexity_score=complexity_score
            )
            
            self.cognitive_load_history.append(metrics)
            return metrics
            
        except Exception as e:
            logging.error(f"[{self.name}] Error assessing cognitive load: {e}")
            return CognitiveLoadMetrics(0, 0, 0, 0, 0, 0)
    
    def should_trigger_reflex(self, input_text: str, context: Dict[str, Any] = None) -> Tuple[bool, ReflexTriggerType]:
        """Determine if reflex agent should handle this input."""
        if not self.reflex_mode_enabled:
            return False, None
        
        context = context or {}
        
        # Check for manual activation
        if context.get('force_reflex', False):
            return True, ReflexTriggerType.MANUAL_ACTIVATION
        
        if not self.auto_trigger_enabled:
            return False, None
        
        # Check for low-effort task
        input_lower = input_text.lower().strip()
        if any(keyword in input_lower for keyword in self.low_effort_keywords):
            return True, ReflexTriggerType.LOW_EFFORT_TASK
        
        # Check cognitive load
        load_metrics = self.assess_cognitive_load()
        total_load = (
            (load_metrics.memory_usage_percent / 100) * 0.2 +
            (load_metrics.error_rate) * 0.3 +
            (load_metrics.complexity_score) * 0.3 +
            (min(load_metrics.response_latency_ms / 1000, 5) / 5) * 0.2
        )
        
        if total_load > self.cognitive_load_threshold:
            return True, ReflexTriggerType.COGNITIVE_LOAD_HIGH
        
        # Check timeout threshold (based on recent response times)
        if load_metrics.response_latency_ms > self.timeout_threshold_ms:
            return True, ReflexTriggerType.TIMEOUT_THRESHOLD
        
        return False, None
    
    def _load_heuristic_patterns(self):
        """Load pre-trained heuristic patterns."""
        # Default heuristic patterns
        default_patterns = {
            "greeting": HeuristicPattern(
                pattern_id="greeting",
                trigger_keywords=["hello", "hi", "hey", "greetings"],
                response_template="Hello! I'm ready to help. What can I do for you?",
                confidence_score=0.9
            ),
            "status_check": HeuristicPattern(
                pattern_id="status_check",
                trigger_keywords=["status", "health", "how are you", "running"],
                response_template="System status: All agents operational. Reflex mode active.",
                confidence_score=0.85
            ),
            "simple_yes_no": HeuristicPattern(
                pattern_id="simple_yes_no",
                trigger_keywords=["yes", "no", "ok", "okay", "sure", "nope"],
                response_template="Acknowledged.",
                confidence_score=0.8
            ),
            "list_request": HeuristicPattern(
                pattern_id="list_request",
                trigger_keywords=["list", "show", "display", "what"],
                response_template="I can quickly show you available options. What specifically would you like to see?",
                confidence_score=0.7
            ),
            "help_request": HeuristicPattern(
                pattern_id="help_request",
                trigger_keywords=["help", "assist", "support", "guide"],
                response_template="I'm here to help! For complex tasks, I can defer to specialized agents. What do you need?",
                confidence_score=0.75
            )
        }
        
        self.heuristic_patterns.update(default_patterns)
        
        # Try to load patterns from file if available
        try:
            pattern_file = self.reflex_config.get('pattern_file', 'output/reflex_patterns.json')
            import os
            if os.path.exists(pattern_file):
                with open(pattern_file, 'r') as f:
                    stored_patterns = json.load(f)
                    for pattern_data in stored_patterns:
                        pattern = HeuristicPattern(**pattern_data)
                        self.heuristic_patterns[pattern.pattern_id] = pattern
                logging.info(f"[{self.name}] Loaded {len(stored_patterns)} stored heuristic patterns")
        except Exception as e:
            logging.warning(f"[{self.name}] Could not load stored patterns: {e}")
    
    def _save_heuristic_patterns(self):
        """Save heuristic patterns to file."""
        try:
            pattern_file = self.reflex_config.get('pattern_file', 'output/reflex_patterns.json')
            os.makedirs(os.path.dirname(pattern_file), exist_ok=True)
            
            patterns_data = []
            for pattern in self.heuristic_patterns.values():
                pattern_dict = {
                    'pattern_id': pattern.pattern_id,
                    'trigger_keywords': pattern.trigger_keywords,
                    'response_template': pattern.response_template,
                    'confidence_score': pattern.confidence_score,
                    'usage_count': pattern.usage_count,
                    'success_rate': pattern.success_rate,
                    'context_requirements': pattern.context_requirements
                }
                patterns_data.append(pattern_dict)
            
            with open(pattern_file, 'w') as f:
                json.dump(patterns_data, f, indent=2)
                
        except Exception as e:
            logging.error(f"[{self.name}] Error saving heuristic patterns: {e}")
    
    def _shallow_memory_scan(self, query: str, limit: int = None) -> List[Dict[str, Any]]:
        """Perform shallow memory scan for quick context retrieval."""
        limit = limit or self.shallow_memory_limit
        
        try:
            # Get memory system
            if not self._memory_system:
                from storage.phase2_cognitive_system import Phase2AutonomousCognitiveSystem
                self._memory_system = Phase2AutonomousCognitiveSystem()
            
            # Quick memory search with limited results
            query_lower = query.lower()
            recent_facts = self._memory_system.get_recent_memories(limit=limit//2)
            
            # Simple keyword matching for speed
            relevant_memories = []
            query_words = set(query_lower.split())
            
            for fact in recent_facts:
                if hasattr(fact, 'text'):
                    fact_words = set(fact.text.lower().split())
                    if query_words.intersection(fact_words):
                        relevant_memories.append({
                            'text': fact.text,
                            'confidence': len(query_words.intersection(fact_words)) / len(query_words),
                            'timestamp': getattr(fact, 'timestamp', datetime.now())
                        })
            
            # Sort by relevance and recency
            relevant_memories.sort(key=lambda x: (x['confidence'], x['timestamp']), reverse=True)
            return relevant_memories[:limit//2]
            
        except Exception as e:
            logging.error(f"[{self.name}] Error in shallow memory scan: {e}")
            return []
    
    def _match_heuristic_pattern(self, input_text: str) -> Optional[Tuple[HeuristicPattern, float]]:
        """Match input against heuristic patterns."""
        input_lower = input_text.lower()
        best_match = None
        best_score = 0
        
        for pattern in self.heuristic_patterns.values():
            # Count keyword matches
            matches = sum(1 for keyword in pattern.trigger_keywords if keyword in input_lower)
            if matches > 0:
                # Calculate match score
                match_ratio = matches / len(pattern.trigger_keywords)
                recency_bonus = 0.1 if pattern.last_used and (datetime.now() - pattern.last_used).days < 1 else 0
                success_bonus = (pattern.success_rate - 0.5) * 0.2
                
                total_score = match_ratio * pattern.confidence_score + recency_bonus + success_bonus
                
                if total_score > best_score:
                    best_score = total_score
                    best_match = pattern
        
        return (best_match, best_score) if best_match and best_score > 0.3 else None
    
    def _check_response_cache(self, input_text: str) -> Optional[ReflexAction]:
        """Check if we have a cached response for similar input."""
        # Simple cache lookup based on exact match or high similarity
        input_key = input_text.lower().strip()
        
        if input_key in self.response_cache:
            cached_action = self.response_cache[input_key]
            # Check if cache is still fresh (within last hour)
            if (datetime.now() - cached_action.timestamp).seconds < 3600:
                return cached_action
        
        # Check for similar inputs in recent actions
        for action in list(self.recent_actions)[-10:]:
            if hasattr(action, 'original_input'):
                similarity = self._calculate_text_similarity(input_text, action.original_input)
                if similarity > 0.8:
                    return action
        
        return None
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _should_defer_to_deep_planning(self, input_text: str, context: Dict[str, Any] = None) -> bool:
        """Determine if the task should be deferred to deep planning."""
        context = context or {}
        
        # Check complexity indicators
        complex_keywords = [
            'implement', 'design', 'architecture', 'debug', 'analyze', 'optimize',
            'refactor', 'research', 'investigate', 'plan', 'strategy', 'algorithm'
        ]
        
        input_lower = input_text.lower()
        complexity_score = sum(1 for keyword in complex_keywords if keyword in input_lower)
        
        # Check length - longer inputs usually need deeper processing
        length_factor = min(len(input_text) / 200, 1.0)  # Normalize to 0-1
        
        # Check if it's a question that needs reasoning
        question_indicators = ['why', 'how', 'what if', 'explain', 'compare', 'analyze']
        has_complex_question = any(indicator in input_lower for indicator in question_indicators)
        
        # Calculate defer score
        defer_score = (
            (complexity_score / len(complex_keywords)) * 0.4 +
            length_factor * 0.3 +
            (1.0 if has_complex_question else 0.0) * 0.3
        )
        
        return defer_score > self.defer_threshold
    
    def _create_defer_response(self, trigger_type: ReflexTriggerType) -> ReflexAction:
        """Create a response that defers to deep planning."""
        defer_messages = {
            ReflexTriggerType.COGNITIVE_LOAD_HIGH: "System load is high. Deferring to specialized planning agents for optimal processing.",
            ReflexTriggerType.TIMEOUT_THRESHOLD: "This request requires deeper analysis. Routing to planning system for comprehensive response.",
            ReflexTriggerType.LOW_EFFORT_TASK: "This appears to need more detailed consideration. Transferring to full planning system.",
            ReflexTriggerType.MANUAL_ACTIVATION: "Complexity detected. Deferring to deep planning for thorough analysis."
        }
        
        message = defer_messages.get(trigger_type, "Deferring to deep planning system for detailed processing.")
        
        return ReflexAction(
            action_type=ReflexResponse.DEFER_TO_DEEP,
            response_text=message,
            confidence=0.9,
            trigger_type=trigger_type,
            processing_time_ms=10,  # Very fast defer decision
            deferred_to_deep=True
        )
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """
        Generate rapid reflex response or defer to deep planning.
        """
        start_time = time.time()
        context = context or {}
        
        try:
            # Check if we should trigger reflex response
            should_trigger, trigger_type = self.should_trigger_reflex(message, context)
            
            if not should_trigger:
                return self._create_defer_response(ReflexTriggerType.MANUAL_ACTIVATION).response_text
            
            # Check if we should defer to deep planning immediately
            if self._should_defer_to_deep_planning(message, context):
                defer_action = self._create_defer_response(trigger_type)
                self.recent_actions.append(defer_action)
                return defer_action.response_text
            
            # Try cached response first
            cached_response = self._check_response_cache(message)
            if cached_response:
                cached_response.trigger_type = trigger_type
                self.recent_actions.append(cached_response)
                return cached_response.response_text
            
            # Try heuristic pattern matching
            pattern_match = self._match_heuristic_pattern(message)
            if pattern_match:
                pattern, confidence = pattern_match
                
                # Update pattern usage
                pattern.usage_count += 1
                pattern.last_used = datetime.now()
                
                # Create contextual response
                response_text = pattern.response_template
                
                # Add shallow memory context if relevant
                if confidence > 0.7:
                    memory_context = self._shallow_memory_scan(message, limit=3)
                    if memory_context:
                        context_summary = "; ".join([mem['text'][:50] + "..." for mem in memory_context[:2]])
                        response_text += f" (Context: {context_summary})"
                
                processing_time = (time.time() - start_time) * 1000
                
                action = ReflexAction(
                    action_type=ReflexResponse.HEURISTIC_MATCH,
                    response_text=response_text,
                    confidence=confidence,
                    trigger_type=trigger_type,
                    processing_time_ms=processing_time,
                    heuristic_used=pattern.pattern_id
                )
                
                # Store original input for cache
                action.original_input = message
                
                self.recent_actions.append(action)
                self.response_cache[message.lower().strip()] = action
                
                return response_text
            
            # If no pattern matches and within time limit, generate simple response
            processing_time = (time.time() - start_time) * 1000
            if processing_time < self.max_reflex_time_ms:
                simple_response = self._generate_simple_response(message, trigger_type)
                action = ReflexAction(
                    action_type=ReflexResponse.IMMEDIATE_ACTION,
                    response_text=simple_response,
                    confidence=0.6,
                    trigger_type=trigger_type,
                    processing_time_ms=processing_time
                )
                
                self.recent_actions.append(action)
                return simple_response
            
            # Exceeded time limit - defer to deep planning
            defer_action = self._create_defer_response(trigger_type)
            self.recent_actions.append(defer_action)
            return defer_action.response_text
            
        except Exception as e:
            logging.error(f"[{self.name}] Error in respond: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            error_action = ReflexAction(
                action_type=ReflexResponse.DEFER_TO_DEEP,
                response_text="Encountered an issue in reflex processing. Deferring to deep planning system.",
                confidence=0.5,
                trigger_type=trigger_type or ReflexTriggerType.MANUAL_ACTIVATION,
                processing_time_ms=processing_time,
                deferred_to_deep=True
            )
            
            self.recent_actions.append(error_action)
            return error_action.response_text
    
    def _generate_simple_response(self, message: str, trigger_type: ReflexTriggerType) -> str:
        """Generate a simple response for unmatched patterns."""
        message_lower = message.lower()
        
        # Simple response based on trigger type
        if trigger_type == ReflexTriggerType.LOW_EFFORT_TASK:
            if any(word in message_lower for word in ['yes', 'ok', 'sure']):
                return "Understood."
            elif any(word in message_lower for word in ['no', 'nope', 'cancel']):
                return "Noted. Task cancelled."
            elif 'thanks' in message_lower or 'thank you' in message_lower:
                return "You're welcome!"
        
        # Default responses based on patterns
        if '?' in message:
            return "I can provide a quick response or route this to specialized agents for detailed analysis. How would you like to proceed?"
        elif any(word in message_lower for word in ['do', 'can', 'will']):
            return "I can handle simple tasks quickly. For complex operations, I'll connect you with specialized agents."
        else:
            return "I've processed your input. Let me know if you need immediate assistance or deeper analysis."
    
    def get_reflex_status(self) -> Dict[str, Any]:
        """Get current status of the reflex agent."""
        load_metrics = self.assess_cognitive_load()
        
        recent_response_times = [action.processing_time_ms for action in list(self.recent_actions)[-10:]]
        avg_response_time = sum(recent_response_times) / len(recent_response_times) if recent_response_times else 0
        
        defer_count = sum(1 for action in list(self.recent_actions)[-20:] if action.deferred_to_deep)
        defer_rate = defer_count / min(20, len(self.recent_actions)) if self.recent_actions else 0
        
        return {
            "reflex_mode_enabled": self.reflex_mode_enabled,
            "auto_trigger_enabled": self.auto_trigger_enabled,
            "cognitive_load": {
                "active_agents": load_metrics.active_agents,
                "pending_tasks": load_metrics.pending_tasks,
                "memory_usage_percent": load_metrics.memory_usage_percent,
                "response_latency_ms": load_metrics.response_latency_ms,
                "error_rate": load_metrics.error_rate,
                "complexity_score": load_metrics.complexity_score
            },
            "performance": {
                "total_responses": len(self.recent_actions),
                "avg_response_time_ms": avg_response_time,
                "defer_rate": defer_rate,
                "cached_patterns": len(self.heuristic_patterns),
                "cache_size": len(self.response_cache)
            },
            "thresholds": {
                "cognitive_load_threshold": self.cognitive_load_threshold,
                "timeout_threshold_ms": self.timeout_threshold_ms,
                "defer_threshold": self.defer_threshold,
                "max_reflex_time_ms": self.max_reflex_time_ms
            }
        }
    
    def learn_from_feedback(self, original_input: str, feedback: str, success: bool):
        """Learn from user feedback to improve heuristic patterns."""
        try:
            # Find the pattern that was used for this input
            pattern_match = self._match_heuristic_pattern(original_input)
            if pattern_match:
                pattern, _ = pattern_match
                
                # Update success rate
                if success:
                    pattern.success_rate = min(1.0, pattern.success_rate * 0.9 + 0.1)
                else:
                    pattern.success_rate = max(0.1, pattern.success_rate * 0.9)
                
                # Record in success rate window
                self.success_rate_window.append(1.0 if success else 0.0)
                
                logging.info(f"[{self.name}] Updated pattern '{pattern.pattern_id}' success rate: {pattern.success_rate:.2f}")
            
            # Save updated patterns
            self._save_heuristic_patterns()
            
        except Exception as e:
            logging.error(f"[{self.name}] Error learning from feedback: {e}")
    
    def get_autonomous_planner(self):
        """Get reference to autonomous planner for integration."""
        if not self._autonomous_planner:
            try:
                from agents.registry import AgentRegistry
                registry = AgentRegistry()
                self._autonomous_planner = registry.get_agent('autonomous_planner')
            except Exception as e:
                logging.error(f"[{self.name}] Could not get autonomous planner: {e}")
        
        return self._autonomous_planner
    
    def signal_defer_to_deep_plan(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Signal the autonomous planner to handle complex request."""
        try:
            planner = self.get_autonomous_planner()
            if planner:
                # Create a planning request
                planning_context = {
                    'source': 'fast_reflex_agent',
                    'reason': 'complexity_exceeds_reflex_capability',
                    'original_message': message,
                    'reflex_context': context or {}
                }
                
                # Use the dedicated defer handler if available, otherwise use respond
                if hasattr(planner, 'handle_reflex_defer_signal'):
                    result = planner.handle_reflex_defer_signal(message, planning_context)
                else:
                    result = planner.respond(message, planning_context)
                
                return {
                    "status": "deferred_to_deep_plan",
                    "planner_response": result,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "error",
                    "message": "Autonomous planner not available for deep planning"
                }
                
        except Exception as e:
            logging.error(f"[{self.name}] Error deferring to deep plan: {e}")
            return {
                "status": "error",
                "message": f"Failed to defer to deep planning: {str(e)}"
            }