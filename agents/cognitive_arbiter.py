#!/usr/bin/env python3
"""
Neural Control Hooks - CognitiveArbiter for MeRNSTA Phase 33

Dynamic arbitration system between reflex and deep reasoning systems, based on
context, dissonance, emotion state, and cognitive load. Acts like a nervous 
system routing decisions to appropriate cognitive agents.
"""

import logging
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum
from pathlib import Path

from .base import BaseAgent
from config.settings import get_config


class CognitiveMode(Enum):
    """Available cognitive processing modes"""
    FAST_REFLEX = "fast_reflex"      # Quick, intuitive responses
    PLANNER = "planner"              # Structured planning and analysis  
    META_SELF = "meta_self"          # Deep reasoning and meta-cognition
    HYBRID = "hybrid"                # Combination approach


class ArbitrationReason(Enum):
    """Reasons for arbitration decisions"""
    HIGH_DISSONANCE = "high_dissonance"
    LOW_CONFIDENCE = "low_confidence"
    TIME_PRESSURE = "time_pressure"
    HIGH_STAKES = "high_stakes"
    ROUTINE_TASK = "routine_task"
    COMPLEX_ANALYSIS = "complex_analysis"
    EMOTIONAL_VOLATILITY = "emotional_volatility"
    TRAIT_BASED = "trait_based"
    FALLBACK = "fallback"


@dataclass
class ArbitrationInput:
    """Input data for cognitive arbitration decisions"""
    message: str
    context: Dict[str, Any]
    dissonance_pressure: float  # 0.0-1.0
    confidence_score: float     # 0.0-1.0
    time_budget: float         # seconds available
    trait_profile: Dict[str, float]  # personality traits
    cognitive_load: float      # current system load 0.0-1.0
    task_complexity: float     # estimated complexity 0.0-1.0
    emotional_state: Dict[str, float]  # emotional factors
    timestamp: datetime


@dataclass 
class ArbitrationDecision:
    """Result of cognitive arbitration"""
    chosen_mode: CognitiveMode
    primary_reason: ArbitrationReason
    confidence: float          # confidence in the decision 0.0-1.0
    reasoning_chain: List[str] # step-by-step reasoning
    backup_mode: Optional[CognitiveMode]  # fallback if primary fails
    execution_priority: int    # 1-10 priority level
    expected_duration: float   # estimated execution time
    risk_assessment: Dict[str, float]  # risk factors
    decision_id: str
    timestamp: datetime
    

@dataclass
class ArbitrationTrace:
    """Complete trace of an arbitration decision and execution"""
    decision: ArbitrationDecision
    input_data: ArbitrationInput
    execution_agent: str
    execution_start: datetime
    execution_end: Optional[datetime]
    success: Optional[bool]
    performance_metrics: Dict[str, float]
    feedback_loop: List[str]  # any adjustments made
    

class CognitiveArbiter(BaseAgent):
    """
    Neural Control Hooks - Dynamic arbitration between cognitive systems.
    
    Features:
    - Real-time arbitration between FastReflex/Planner/MetaSelf agents
    - Multi-factor decision analysis (dissonance, confidence, time, traits)
    - Cognitive feedback loops and adaptation
    - Complete audit trails and traces
    - Nervous system-like routing and priority management
    """
    
    def __init__(self):
        super().__init__("cognitive_arbiter")
        
        # Load configuration
        self.config = get_config()
        self.arbiter_config = self.config.get('cognitive_arbiter', {})
        
        # Arbitration thresholds (configurable, no hardcoding per user's memory)
        thresholds = self.arbiter_config.get('thresholds', {})
        self.high_dissonance_threshold = thresholds.get('high_dissonance', 0.7)
        self.low_confidence_threshold = thresholds.get('low_confidence', 0.4)
        self.time_pressure_threshold = thresholds.get('time_pressure_seconds', 10.0)
        self.high_complexity_threshold = thresholds.get('high_complexity', 0.8)
        self.high_stakes_threshold = thresholds.get('high_stakes', 0.9)
        
        # Trait influence weights (configurable)
        trait_weights = self.arbiter_config.get('trait_weights', {})
        self.caution_weight = trait_weights.get('caution', 0.3)
        self.emotional_sensitivity_weight = trait_weights.get('emotional_sensitivity', 0.25)
        self.analytical_preference_weight = trait_weights.get('analytical_preference', 0.2)
        self.speed_preference_weight = trait_weights.get('speed_preference', 0.25)
        
        # Mode selection scoring (configurable)
        mode_scoring = self.arbiter_config.get('mode_scoring', {})
        self.reflex_base_score = mode_scoring.get('reflex_base', 0.6)
        self.planner_base_score = mode_scoring.get('planner_base', 0.7)
        self.meta_self_base_score = mode_scoring.get('meta_self_base', 0.5)
        
        # Agent access
        self._agent_registry = None
        self._dissonance_tracker = None
        self._personality_engine = None
        
        # Decision tracking
        self.decision_history: List[ArbitrationTrace] = []
        self.performance_stats: Dict[CognitiveMode, Dict[str, float]] = {
            mode: {'success_rate': 0.0, 'avg_duration': 0.0, 'total_uses': 0}
            for mode in CognitiveMode
        }
        
        # Feedback learning
        self.adaptation_weights: Dict[str, float] = {
            'success_feedback': 0.1,
            'timing_feedback': 0.05,
            'quality_feedback': 0.08
        }
        
        # Storage for audit trails
        self.trace_storage_path = Path("output/arbitration_traces.jsonl")
        self.trace_storage_path.parent.mkdir(exist_ok=True)
        
        logging.info(f"[{self.name}] Initialized Neural Control Hooks arbitration system")
    
    def get_agent_instructions(self) -> str:
        """Instructions for the CognitiveArbiter agent."""
        return """
        I am the Cognitive Arbiter - the neural control system that routes decisions 
        between fast reflex responses, structured planning, and deep meta-cognition.
        
        I analyze:
        - Cognitive dissonance pressure and urgency
        - Confidence levels and uncertainty
        - Time constraints and cognitive load
        - Personality traits and emotional state
        - Task complexity and stakes
        
        I route to:
        - FastReflexAgent: Quick, intuitive responses for routine tasks
        - PlannerAgent: Structured analysis for complex problems
        - MetaSelfAgent: Deep reasoning for high-stakes decisions
        
        I provide complete audit trails showing why each decision was made.
        """
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """Main arbitration and execution method."""
        if not context:
            context = {}
            
        try:
            # Gather arbitration inputs
            arbitration_input = self._gather_arbitration_inputs(message, context)
            
            # Make arbitration decision
            decision = self._arbitrate_cognitive_mode(arbitration_input)
            
            # Execute the chosen cognitive mode
            result = self._execute_arbitration_decision(decision, arbitration_input)
            
            # Log the complete trace
            self._log_arbitration_trace(decision, arbitration_input, result)
            
            return result['response']
            
        except Exception as e:
            logging.error(f"[{self.name}] Error in arbitration: {e}")
            # Fallback to fast reflex for error conditions
            return self._fallback_to_reflex(message, context)
    
    def _gather_arbitration_inputs(self, message: str, context: Dict[str, Any]) -> ArbitrationInput:
        """Gather all inputs needed for arbitration decision."""
        
        # Get dissonance pressure from DissonanceTracker
        dissonance_pressure = self._get_dissonance_pressure()
        
        # Calculate confidence score from multiple sources
        confidence_score = self._calculate_confidence_score(message, context)
        
        # Determine time budget
        time_budget = context.get('time_budget', self._estimate_time_budget(message, context))
        
        # Get trait profile from personality system
        trait_profile = self._get_trait_profile()
        
        # Assess cognitive load
        cognitive_load = self._assess_cognitive_load()
        
        # Estimate task complexity
        task_complexity = self._estimate_task_complexity(message, context)
        
        # Get emotional state
        emotional_state = self._get_emotional_state()
        
        return ArbitrationInput(
            message=message,
            context=context,
            dissonance_pressure=dissonance_pressure,
            confidence_score=confidence_score,
            time_budget=time_budget,
            trait_profile=trait_profile,
            cognitive_load=cognitive_load,
            task_complexity=task_complexity,
            emotional_state=emotional_state,
            timestamp=datetime.now()
        )
    
    def _arbitrate_cognitive_mode(self, inputs: ArbitrationInput) -> ArbitrationDecision:
        """Core arbitration logic - decide which cognitive mode to use."""
        
        # Calculate scores for each mode
        mode_scores = self._calculate_mode_scores(inputs)
        
        # Determine primary choice
        chosen_mode = max(mode_scores.keys(), key=lambda k: mode_scores[k])
        primary_reason = self._determine_primary_reason(inputs, chosen_mode)
        
        # Calculate decision confidence
        decision_confidence = self._calculate_decision_confidence(mode_scores, inputs)
        
        # Generate reasoning chain
        reasoning_chain = self._generate_reasoning_chain(inputs, mode_scores, chosen_mode)
        
        # Determine backup mode
        backup_mode = self._select_backup_mode(mode_scores, chosen_mode)
        
        # Risk assessment
        risk_assessment = self._assess_decision_risks(inputs, chosen_mode)
        
        # Execution priority (1-10)
        execution_priority = self._calculate_execution_priority(inputs, chosen_mode)
        
        # Expected duration
        expected_duration = self._estimate_execution_duration(inputs, chosen_mode)
        
        decision_id = f"arb_{int(time.time() * 1000)}"
        
        return ArbitrationDecision(
            chosen_mode=chosen_mode,
            primary_reason=primary_reason,
            confidence=decision_confidence,
            reasoning_chain=reasoning_chain,
            backup_mode=backup_mode,
            execution_priority=execution_priority,
            expected_duration=expected_duration,
            risk_assessment=risk_assessment,
            decision_id=decision_id,
            timestamp=datetime.now()
        )
    
    def _calculate_mode_scores(self, inputs: ArbitrationInput) -> Dict[CognitiveMode, float]:
        """Calculate scores for each cognitive mode based on current inputs."""
        
        scores = {
            CognitiveMode.FAST_REFLEX: self.reflex_base_score,
            CognitiveMode.PLANNER: self.planner_base_score,
            CognitiveMode.META_SELF: self.meta_self_base_score
        }
        
        # Dissonance pressure adjustments
        if inputs.dissonance_pressure > self.high_dissonance_threshold:
            # High dissonance favors meta-cognition for resolution
            scores[CognitiveMode.META_SELF] += 0.3
            scores[CognitiveMode.PLANNER] += 0.1
            scores[CognitiveMode.FAST_REFLEX] -= 0.2
        elif inputs.dissonance_pressure < 0.3:
            # Low dissonance allows fast responses
            scores[CognitiveMode.FAST_REFLEX] += 0.2
        
        # Confidence adjustments
        if inputs.confidence_score < self.low_confidence_threshold:
            # Low confidence needs careful analysis
            scores[CognitiveMode.META_SELF] += 0.2
            scores[CognitiveMode.PLANNER] += 0.15
            scores[CognitiveMode.FAST_REFLEX] -= 0.25
        elif inputs.confidence_score > 0.8:
            # High confidence allows reflex responses
            scores[CognitiveMode.FAST_REFLEX] += 0.15
        
        # Time pressure adjustments
        if inputs.time_budget < self.time_pressure_threshold:
            # Time pressure favors quick responses
            scores[CognitiveMode.FAST_REFLEX] += 0.3
            scores[CognitiveMode.PLANNER] -= 0.15
            scores[CognitiveMode.META_SELF] -= 0.25
        elif inputs.time_budget > 120:  # More than 2 minutes
            # Plenty of time allows deep thinking
            scores[CognitiveMode.META_SELF] += 0.2
            scores[CognitiveMode.PLANNER] += 0.1
        
        # Task complexity adjustments
        if inputs.task_complexity > self.high_complexity_threshold:
            # Complex tasks need structured thinking
            scores[CognitiveMode.PLANNER] += 0.25
            scores[CognitiveMode.META_SELF] += 0.15
            scores[CognitiveMode.FAST_REFLEX] -= 0.3
        elif inputs.task_complexity < 0.3:
            # Simple tasks can use reflexes
            scores[CognitiveMode.FAST_REFLEX] += 0.2
        
        # Cognitive load adjustments
        if inputs.cognitive_load > 0.8:
            # High load favors simpler processing
            scores[CognitiveMode.FAST_REFLEX] += 0.2
            scores[CognitiveMode.PLANNER] -= 0.1
            scores[CognitiveMode.META_SELF] -= 0.15
        
        # Trait-based adjustments
        trait_profile = inputs.trait_profile
        
        # Caution trait influences
        caution_level = trait_profile.get('caution', 0.5)
        if caution_level > 0.7:
            scores[CognitiveMode.META_SELF] += self.caution_weight * caution_level
            scores[CognitiveMode.PLANNER] += self.caution_weight * 0.5 * caution_level
            scores[CognitiveMode.FAST_REFLEX] -= self.caution_weight * caution_level
        
        # Emotional sensitivity influences
        emotional_sensitivity = trait_profile.get('emotional_sensitivity', 0.5)
        emotional_volatility = inputs.emotional_state.get('volatility', 0.0)
        if emotional_sensitivity > 0.6 and emotional_volatility > 0.5:
            # High emotional sensitivity with volatility needs careful handling
            scores[CognitiveMode.META_SELF] += self.emotional_sensitivity_weight
            scores[CognitiveMode.FAST_REFLEX] -= self.emotional_sensitivity_weight * 0.8
        
        # Analytical preference
        analytical_pref = trait_profile.get('analytical_preference', 0.5)
        if analytical_pref > 0.7:
            scores[CognitiveMode.PLANNER] += self.analytical_preference_weight * analytical_pref
            scores[CognitiveMode.META_SELF] += self.analytical_preference_weight * 0.5 * analytical_pref
        
        # Speed preference
        speed_pref = trait_profile.get('speed_preference', 0.5)
        if speed_pref > 0.7:
            scores[CognitiveMode.FAST_REFLEX] += self.speed_preference_weight * speed_pref
        
        # Ensure scores stay in reasonable bounds
        for mode in scores:
            scores[mode] = max(0.0, min(1.0, scores[mode]))
        
        return scores
    
    def _execute_arbitration_decision(self, decision: ArbitrationDecision, 
                                    inputs: ArbitrationInput) -> Dict[str, Any]:
        """Execute the arbitration decision by routing to the chosen agent."""
        
        execution_start = datetime.now()
        
        try:
            # Get the appropriate agent
            agent = self._get_cognitive_agent(decision.chosen_mode)
            
            if not agent:
                # Fallback to reflex if agent not available
                logging.warning(f"[{self.name}] Agent for {decision.chosen_mode} not available, falling back to reflex")
                return self._fallback_to_reflex(inputs.message, inputs.context)
            
            # Execute with the chosen agent
            if decision.chosen_mode == CognitiveMode.FAST_REFLEX:
                response = self._execute_fast_reflex(agent, inputs, decision)
            elif decision.chosen_mode == CognitiveMode.PLANNER:
                response = self._execute_planner(agent, inputs, decision)
            elif decision.chosen_mode == CognitiveMode.META_SELF:
                response = self._execute_meta_self(agent, inputs, decision)
            else:
                raise ValueError(f"Unknown cognitive mode: {decision.chosen_mode}")
            
            execution_end = datetime.now()
            execution_duration = (execution_end - execution_start).total_seconds()
            
            # Update performance stats
            self._update_performance_stats(decision.chosen_mode, True, execution_duration)
            
            return {
                'response': response,
                'execution_agent': agent.name,
                'execution_duration': execution_duration,
                'success': True,
                'performance_metrics': {
                    'duration': execution_duration,
                    'mode': decision.chosen_mode.value,
                    'decision_confidence': decision.confidence
                }
            }
            
        except Exception as e:
            logging.error(f"[{self.name}] Error executing {decision.chosen_mode}: {e}")
            
            # Try backup mode if available
            if decision.backup_mode and decision.backup_mode != decision.chosen_mode:
                logging.info(f"[{self.name}] Trying backup mode: {decision.backup_mode}")
                try:
                    backup_agent = self._get_cognitive_agent(decision.backup_mode)
                    if backup_agent:
                        response = backup_agent.respond(inputs.message, inputs.context)
                        return {
                            'response': response,
                            'execution_agent': backup_agent.name,
                            'execution_duration': (datetime.now() - execution_start).total_seconds(),
                            'success': True,
                            'backup_used': True
                        }
                except Exception as backup_error:
                    logging.error(f"[{self.name}] Backup mode {decision.backup_mode} also failed: {backup_error}")
            
            # Final fallback to reflex
            return self._fallback_to_reflex(inputs.message, inputs.context)
    
    def _get_cognitive_agent(self, mode: CognitiveMode) -> Optional[BaseAgent]:
        """Get the agent instance for a cognitive mode."""
        if not self._agent_registry:
            try:
                from agents.registry import get_agent_registry
                self._agent_registry = get_agent_registry()
            except ImportError:
                logging.error(f"[{self.name}] Could not import agent registry")
                return None
        
        if mode == CognitiveMode.FAST_REFLEX:
            return self._agent_registry.get_agent('fast_reflex')
        elif mode == CognitiveMode.PLANNER:
            return self._agent_registry.get_agent('planner')
        elif mode == CognitiveMode.META_SELF:
            return self._agent_registry.get_agent('meta_self')
        else:
            return None
    
    def _execute_fast_reflex(self, agent: BaseAgent, inputs: ArbitrationInput, 
                           decision: ArbitrationDecision) -> str:
        """Execute fast reflex mode with time constraints."""
        # Add arbitration context to help the reflex agent
        context = inputs.context.copy()
        context.update({
            'arbitration_mode': 'fast_reflex',
            'time_budget': inputs.time_budget,
            'confidence_hint': inputs.confidence_score,
            'reasoning': 'Quick intuitive response requested'
        })
        
        return agent.respond(inputs.message, context)
    
    def _execute_planner(self, agent: BaseAgent, inputs: ArbitrationInput, 
                        decision: ArbitrationDecision) -> str:
        """Execute structured planning mode."""
        context = inputs.context.copy()
        context.update({
            'arbitration_mode': 'planner',
            'complexity_level': inputs.task_complexity,
            'dissonance_pressure': inputs.dissonance_pressure,
            'reasoning': 'Structured analysis requested',
            'expected_depth': 'medium'
        })
        
        return agent.respond(inputs.message, context)
    
    def _execute_meta_self(self, agent: BaseAgent, inputs: ArbitrationInput, 
                          decision: ArbitrationDecision) -> str:
        """Execute meta-cognitive deep reasoning mode."""
        context = inputs.context.copy()
        context.update({
            'arbitration_mode': 'meta_self',
            'dissonance_pressure': inputs.dissonance_pressure,
            'complexity_level': inputs.task_complexity,
            'trait_profile': inputs.trait_profile,
            'reasoning': 'Deep meta-cognitive analysis requested',
            'expected_depth': 'deep'
        })
        
        return agent.respond(inputs.message, context)
    
    def _fallback_to_reflex(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to fast reflex when other modes fail."""
        try:
            reflex_agent = self._get_cognitive_agent(CognitiveMode.FAST_REFLEX)
            if reflex_agent:
                response = reflex_agent.respond(message, context)
                return {
                    'response': response,
                    'execution_agent': reflex_agent.name,
                    'execution_duration': 0.1,
                    'success': True,
                    'fallback_used': True
                }
        except:
            pass
        
        # Ultimate fallback - simple response
        return {
            'response': "I need to process this request, but I'm having difficulty with cognitive routing. Let me try a simpler approach.",
            'execution_agent': 'fallback',
            'execution_duration': 0.0,
            'success': False
        }
    
    def get_arbitration_trace(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent arbitration traces for analysis."""
        recent_traces = self.decision_history[-limit:] if limit > 0 else self.decision_history
        
        traces = []
        for trace in recent_traces:
            traces.append({
                'decision_id': trace.decision.decision_id,
                'timestamp': trace.decision.timestamp.isoformat(),
                'chosen_mode': trace.decision.chosen_mode.value,
                'primary_reason': trace.decision.primary_reason.value,
                'confidence': trace.decision.confidence,
                'success': trace.success,
                'execution_agent': trace.execution_agent,
                'duration': (trace.execution_end - trace.execution_start).total_seconds() if trace.execution_end else None,
                'reasoning_chain': trace.decision.reasoning_chain,
                'input_summary': {
                    'dissonance_pressure': trace.input_data.dissonance_pressure,
                    'confidence_score': trace.input_data.confidence_score,
                    'time_budget': trace.input_data.time_budget,
                    'task_complexity': trace.input_data.task_complexity
                }
            })
        
        return traces
    
    # Helper methods for gathering inputs
    def _get_dissonance_pressure(self) -> float:
        """Get current dissonance pressure from DissonanceTracker."""
        try:
            if not self._dissonance_tracker:
                from agents.registry import get_agent_registry
                registry = get_agent_registry()
                self._dissonance_tracker = registry.get_agent('dissonance_tracker')
            
            if self._dissonance_tracker:
                # Get average pressure from active dissonance regions
                regions = self._dissonance_tracker.dissonance_regions
                if regions:
                    pressure_scores = [region.pressure_score for region in regions.values()]
                    return sum(pressure_scores) / len(pressure_scores)
        except Exception as e:
            logging.debug(f"[{self.name}] Could not get dissonance pressure: {e}")
        
        return 0.3  # Default moderate pressure
    
    def _calculate_confidence_score(self, message: str, context: Dict[str, Any]) -> float:
        """Calculate overall confidence score from multiple sources."""
        # Start with base confidence
        confidence = 0.6
        
        # Check context for confidence hints
        if 'confidence' in context:
            confidence = context['confidence']
        
        # Add message analysis
        uncertainty_indicators = ['maybe', 'perhaps', 'might', 'could', 'unsure', 'don\'t know']
        message_lower = message.lower()
        uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in message_lower)
        confidence -= uncertainty_count * 0.1
        
        # Confidence from message length and complexity
        word_count = len(message.split())
        if word_count < 5:
            confidence -= 0.1  # Very short messages are often uncertain
        elif word_count > 50:
            confidence += 0.1  # Detailed messages often show confidence
        
        return max(0.0, min(1.0, confidence))
    
    def _estimate_time_budget(self, message: str, context: Dict[str, Any]) -> float:
        """Estimate available time budget for processing."""
        # Check context for explicit time budget
        if 'time_budget' in context:
            return context['time_budget']
        
        # Estimate based on message urgency indicators
        urgent_words = ['urgent', 'quickly', 'asap', 'immediately', 'now', 'fast']
        message_lower = message.lower()
        
        if any(word in message_lower for word in urgent_words):
            return 5.0  # 5 seconds for urgent requests
        
        # Default reasonable time budget
        return 30.0  # 30 seconds
    
    def _get_trait_profile(self) -> Dict[str, float]:
        """Get personality trait profile."""
        try:
            if not self._personality_engine:
                from agents.registry import get_agent_registry
                registry = get_agent_registry()
                self._personality_engine = registry.get_agent('personality_engine')
            
            if self._personality_engine:
                # Get current personality traits
                return getattr(self._personality_engine, 'trait_profile', {})
        except Exception as e:
            logging.debug(f"[{self.name}] Could not get trait profile: {e}")
        
        # Default balanced trait profile
        return {
            'caution': 0.5,
            'emotional_sensitivity': 0.5,
            'analytical_preference': 0.6,
            'speed_preference': 0.4
        }
    
    def _assess_cognitive_load(self) -> float:
        """Assess current cognitive load on the system."""
        # This could integrate with system monitoring
        # For now, use a simple heuristic
        active_traces = len(self.decision_history)
        recent_decisions = len([t for t in self.decision_history 
                              if (datetime.now() - t.decision.timestamp).total_seconds() < 60])
        
        # Higher recent activity = higher cognitive load
        load = min(1.0, recent_decisions / 10.0)
        return load
    
    def _estimate_task_complexity(self, message: str, context: Dict[str, Any]) -> float:
        """Estimate task complexity from message and context."""
        complexity = 0.3  # Base complexity
        
        # Message length factor
        word_count = len(message.split())
        complexity += min(0.3, word_count / 100.0)
        
        # Complexity indicators
        complex_words = ['analyze', 'evaluate', 'compare', 'synthesize', 'research', 
                        'investigate', 'design', 'architect', 'optimize']
        message_lower = message.lower()
        complexity_indicators = sum(1 for word in complex_words if word in message_lower)
        complexity += complexity_indicators * 0.1
        
        # Context complexity
        if context.get('requires_analysis', False):
            complexity += 0.2
        if context.get('multiple_steps', False):
            complexity += 0.2
        
        return min(1.0, complexity)
    
    def _get_emotional_state(self) -> Dict[str, float]:
        """Get current emotional state indicators."""
        # This could integrate with emotion tracking systems
        # For now, return a neutral state
        return {
            'volatility': 0.2,
            'stress': 0.3,
            'excitement': 0.4,
            'confidence': 0.6
        }
    
    def _determine_primary_reason(self, inputs: ArbitrationInput, chosen_mode: CognitiveMode) -> ArbitrationReason:
        """Determine the primary reason for the arbitration decision."""
        if inputs.dissonance_pressure > self.high_dissonance_threshold:
            return ArbitrationReason.HIGH_DISSONANCE
        elif inputs.confidence_score < self.low_confidence_threshold:
            return ArbitrationReason.LOW_CONFIDENCE
        elif inputs.time_budget < self.time_pressure_threshold:
            return ArbitrationReason.TIME_PRESSURE
        elif inputs.task_complexity > self.high_complexity_threshold:
            return ArbitrationReason.COMPLEX_ANALYSIS
        elif inputs.emotional_state.get('volatility', 0) > 0.7:
            return ArbitrationReason.EMOTIONAL_VOLATILITY
        elif inputs.trait_profile.get('caution', 0.5) > 0.8:
            return ArbitrationReason.TRAIT_BASED
        elif inputs.task_complexity < 0.3:
            return ArbitrationReason.ROUTINE_TASK
        else:
            return ArbitrationReason.FALLBACK
    
    def _calculate_decision_confidence(self, mode_scores: Dict[CognitiveMode, float], 
                                     inputs: ArbitrationInput) -> float:
        """Calculate confidence in the arbitration decision."""
        # Base confidence from score spread
        sorted_scores = sorted(mode_scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            score_spread = sorted_scores[0] - sorted_scores[1]
            confidence = 0.5 + score_spread  # Higher spread = more confidence
        else:
            confidence = 0.7
        
        # Adjust based on input quality
        input_quality = (inputs.confidence_score + (1 - inputs.dissonance_pressure)) / 2
        confidence = (confidence + input_quality) / 2
        
        return max(0.1, min(1.0, confidence))
    
    def _generate_reasoning_chain(self, inputs: ArbitrationInput, 
                                mode_scores: Dict[CognitiveMode, float], 
                                chosen_mode: CognitiveMode) -> List[str]:
        """Generate step-by-step reasoning for the decision."""
        chain = []
        
        # Input assessment
        chain.append(f"Input assessment: dissonance={inputs.dissonance_pressure:.2f}, confidence={inputs.confidence_score:.2f}, time_budget={inputs.time_budget:.1f}s")
        
        # Key decision factors
        if inputs.dissonance_pressure > self.high_dissonance_threshold:
            chain.append(f"High dissonance pressure ({inputs.dissonance_pressure:.2f}) detected - favoring deeper analysis")
        
        if inputs.confidence_score < self.low_confidence_threshold:
            chain.append(f"Low confidence ({inputs.confidence_score:.2f}) - need careful reasoning")
        
        if inputs.time_budget < self.time_pressure_threshold:
            chain.append(f"Time pressure ({inputs.time_budget:.1f}s) - favoring quick response")
        
        if inputs.task_complexity > self.high_complexity_threshold:
            chain.append(f"High task complexity ({inputs.task_complexity:.2f}) - structured approach needed")
        
        # Trait influences
        trait_influences = []
        for trait, value in inputs.trait_profile.items():
            if value > 0.7:
                trait_influences.append(f"{trait}={value:.2f}")
        if trait_influences:
            chain.append(f"Trait influences: {', '.join(trait_influences)}")
        
        # Score summary
        score_summary = ", ".join([f"{mode.value}={score:.2f}" for mode, score in mode_scores.items()])
        chain.append(f"Mode scores: {score_summary}")
        
        # Final decision
        chain.append(f"Chosen: {chosen_mode.value} (score={mode_scores[chosen_mode]:.2f})")
        
        return chain
    
    def _select_backup_mode(self, mode_scores: Dict[CognitiveMode, float], 
                          chosen_mode: CognitiveMode) -> Optional[CognitiveMode]:
        """Select backup mode in case primary fails."""
        # Get second highest scoring mode
        sorted_modes = sorted(mode_scores.items(), key=lambda x: x[1], reverse=True)
        
        for mode, score in sorted_modes:
            if mode != chosen_mode and score > 0.3:  # Minimum viable score
                return mode
        
        # Default fallback is always fast reflex
        return CognitiveMode.FAST_REFLEX if chosen_mode != CognitiveMode.FAST_REFLEX else None
    
    def _assess_decision_risks(self, inputs: ArbitrationInput, chosen_mode: CognitiveMode) -> Dict[str, float]:
        """Assess risks associated with the decision."""
        risks = {}
        
        # Time risk
        if inputs.time_budget < 5 and chosen_mode != CognitiveMode.FAST_REFLEX:
            risks['time_overrun'] = 0.8
        
        # Quality risk 
        if inputs.task_complexity > 0.8 and chosen_mode == CognitiveMode.FAST_REFLEX:
            risks['inadequate_analysis'] = 0.7
        
        # Confidence risk
        if inputs.confidence_score < 0.3:
            risks['poor_quality'] = 0.6
        
        # Dissonance risk
        if inputs.dissonance_pressure > 0.8 and chosen_mode != CognitiveMode.META_SELF:
            risks['unresolved_conflicts'] = 0.7
        
        return risks
    
    def _calculate_execution_priority(self, inputs: ArbitrationInput, chosen_mode: CognitiveMode) -> int:
        """Calculate execution priority (1-10)."""
        priority = 5  # Base priority
        
        # Time pressure increases priority
        if inputs.time_budget < 10:
            priority += 3
        elif inputs.time_budget < 30:
            priority += 1
        
        # High dissonance increases priority
        if inputs.dissonance_pressure > 0.8:
            priority += 2
        
        # High complexity with deep mode increases priority
        if inputs.task_complexity > 0.8 and chosen_mode == CognitiveMode.META_SELF:
            priority += 1
        
        return max(1, min(10, priority))
    
    def _estimate_execution_duration(self, inputs: ArbitrationInput, chosen_mode: CognitiveMode) -> float:
        """Estimate how long execution will take."""
        base_durations = {
            CognitiveMode.FAST_REFLEX: 2.0,
            CognitiveMode.PLANNER: 15.0,
            CognitiveMode.META_SELF: 45.0
        }
        
        base_duration = base_durations.get(chosen_mode, 10.0)
        
        # Adjust for complexity
        complexity_factor = 1 + (inputs.task_complexity * 0.5)
        
        # Adjust for message length
        word_count = len(inputs.message.split())
        length_factor = 1 + min(0.5, word_count / 100.0)
        
        return base_duration * complexity_factor * length_factor
    
    def _update_performance_stats(self, mode: CognitiveMode, success: bool, duration: float):
        """Update performance statistics for learning."""
        stats = self.performance_stats[mode]
        
        # Update total uses
        stats['total_uses'] += 1
        
        # Update success rate (running average)
        current_rate = stats['success_rate']
        stats['success_rate'] = (current_rate * (stats['total_uses'] - 1) + (1.0 if success else 0.0)) / stats['total_uses']
        
        # Update average duration (running average) 
        current_duration = stats['avg_duration']
        stats['avg_duration'] = (current_duration * (stats['total_uses'] - 1) + duration) / stats['total_uses']
    
    def _log_arbitration_trace(self, decision: ArbitrationDecision, inputs: ArbitrationInput, 
                             execution_result: Dict[str, Any]):
        """Log complete arbitration trace for audit and learning."""
        
        trace = ArbitrationTrace(
            decision=decision,
            input_data=inputs,
            execution_agent=execution_result.get('execution_agent', 'unknown'),
            execution_start=decision.timestamp,
            execution_end=datetime.now(),
            success=execution_result.get('success', False),
            performance_metrics=execution_result.get('performance_metrics', {}),
            feedback_loop=[]
        )
        
        # Add to memory
        self.decision_history.append(trace)
        
        # Keep only recent traces in memory
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]
        
        # Save to persistent storage
        try:
            with open(self.trace_storage_path, 'a') as f:
                trace_data = {
                    'decision_id': decision.decision_id,
                    'timestamp': decision.timestamp.isoformat(),
                    'chosen_mode': decision.chosen_mode.value,
                    'primary_reason': decision.primary_reason.value,
                    'confidence': decision.confidence,
                    'reasoning_chain': decision.reasoning_chain,
                    'input_data': {
                        'dissonance_pressure': inputs.dissonance_pressure,
                        'confidence_score': inputs.confidence_score,
                        'time_budget': inputs.time_budget,
                        'task_complexity': inputs.task_complexity,
                        'cognitive_load': inputs.cognitive_load
                    },
                    'execution_result': execution_result,
                    'performance_stats': dict(self.performance_stats)
                }
                f.write(json.dumps(trace_data) + '\n')
        except Exception as e:
            logging.error(f"[{self.name}] Error saving trace: {e}")
        
        logging.info(f"[{self.name}] Decision {decision.decision_id}: {decision.chosen_mode.value} "
                    f"(reason: {decision.primary_reason.value}, confidence: {decision.confidence:.2f})")