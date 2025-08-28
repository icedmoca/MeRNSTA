#!/usr/bin/env python3
"""
ReflectionOrchestrator - Coordinated Self-Reflection and Debate Triggering for MeRNSTA Phase 21

Coordinates debate as part of internal self-reflection, triggering debates when contradictions arise,
uncertainty thresholds are high, or ethical concerns are detected.
"""

import logging
import uuid
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

from .base import BaseAgent
from config.settings import get_config


class ReflectionTrigger(Enum):
    """Types of triggers for reflection."""
    CONTRADICTION = "contradiction"
    UNCERTAINTY = "uncertainty"
    ETHICAL_CONCERN = "ethical_concern"
    PERFORMANCE_ISSUE = "performance_issue"
    CONFLICTING_GOALS = "conflicting_goals"
    FEEDBACK_ANALYSIS = "feedback_analysis"
    PERIODIC_REVIEW = "periodic_review"
    REPLICATION_NEEDED = "replication_needed"


class ReflectionPriority(Enum):
    """Priority levels for reflection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ReflectionRequest:
    """Request for reflection on a topic."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trigger: ReflectionTrigger = ReflectionTrigger.PERIODIC_REVIEW
    priority: ReflectionPriority = ReflectionPriority.MEDIUM
    topic: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    memory_refs: List[str] = field(default_factory=list)
    resolution_required: bool = True


@dataclass
class ReflectionResult:
    """Result of a reflection process."""
    request_id: str
    debate_triggered: bool
    debate_id: Optional[str] = None
    insights: List[str] = field(default_factory=list)
    contradictions_identified: List[str] = field(default_factory=list)
    belief_updates: List[Dict[str, Any]] = field(default_factory=list)
    action_recommendations: List[str] = field(default_factory=list)
    confidence_change: float = 0.0
    resolved: bool = False
    follow_up_needed: bool = False


class ReflectionOrchestrator(BaseAgent):
    """
    Coordinates debate as part of internal self-reflection.
    
    Capabilities:
    - Automatic contradiction detection
    - Uncertainty threshold monitoring  
    - Ethical concern identification
    - Debate triggering and coordination
    - Belief system updates
    - Learning from reflection outcomes
    """
    
    def __init__(self):
        super().__init__("reflection_orchestrator")
        
        # Load reflection configuration
        self.config = self._load_reflection_config()
        self.enabled = self.config.get('enabled', True)
        
        # Thresholds for triggering reflection
        reflection_trigger = self.config.get('reflection_trigger', {})
        self.contradiction_threshold = reflection_trigger.get('contradiction_threshold', 0.6)
        self.uncertainty_threshold = reflection_trigger.get('uncertainty_threshold', 0.7)
        self.ethical_threshold = reflection_trigger.get('ethical_threshold', 0.5)
        self.performance_threshold = reflection_trigger.get('performance_threshold', 0.4)
        
        # Reflection management
        self.pending_reflections: List[ReflectionRequest] = []
        self.active_reflections: Dict[str, Dict[str, Any]] = {}
        self.reflection_history: List[ReflectionResult] = []
        
        # Performance tracking
        self.contradiction_count = 0
        self.uncertainty_events = 0
        self.ethical_flags = 0
        self.last_reflection_time = None
        
        # Integration with other systems
        self.debate_engine = None
        self.memory_analyzer = None
        
        # Phase 22: Initialize agent replicator
        self.agent_replicator = None
        self.replication_config = get_config().get('self_replication', {}).get('agent_replication', {})
        self.replication_enabled = self.replication_config.get('enabled', False)
        
        # Replication thresholds
        self.replication_contradiction_threshold = 5  # Number of contradictions to trigger replication
        self.replication_uncertainty_threshold = 0.8  # Uncertainty level to trigger replication
        self.replication_failure_threshold = 3      # Consecutive failures to trigger replication
        self.min_replication_interval = 3600        # Minimum seconds between replications
        self.last_replication_time = None
        
        # Performance tracking for replication
        self.consecutive_failures = 0
        self.performance_decline_counter = 0
        
        # Initialize agent replicator if enabled
        if self.replication_enabled:
            try:
                from .self_replicator import get_agent_replicator
                self.agent_replicator = get_agent_replicator()
                logging.info(f"[{self.name}] Agent replicator initialized for performance-driven evolution")
            except ImportError:
                logging.warning(f"[{self.name}] Agent replicator not available")
                self.replication_enabled = False
        
        # Initialize memory consolidator for Phase 22
        self.memory_consolidator = None
        try:
            from storage.memory_consolidator import MemoryConsolidator
            self.memory_consolidator = MemoryConsolidator()
            self.consolidation_config = self.config.get('memory_consolidation', {})
            self.last_consolidation_time = None
            self.consolidation_interval = self.consolidation_config.get('auto_consolidation_interval', 3600)
            logging.info(f"[{self.name}] Memory consolidator initialized")
        except ImportError:
            logging.warning(f"[{self.name}] Memory consolidator not available")
        
        logging.info(f"[{self.name}] Initialized reflection orchestrator with thresholds: "
                    f"contradiction={self.contradiction_threshold}, uncertainty={self.uncertainty_threshold}")
    
    def get_agent_instructions(self) -> str:
        """Return specialized instructions for the reflection orchestrator."""
        return (
            "You are the Reflection Orchestrator, responsible for coordinating internal self-reflection "
            "and triggering debates when contradictions, uncertainties, or ethical concerns arise. "
            "Your role is to monitor the system's cognitive state, identify when deeper reflection "
            "is needed, orchestrate debate processes, and integrate insights back into the belief system. "
            "Focus on maintaining cognitive coherence, resolving contradictions, and continuous learning "
            "through structured self-examination and dialectical reasoning."
        )
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """Generate reflection orchestration responses."""
        context = context or {}
        
        # Build memory context for reflection patterns
        memory_context = self.get_memory_context(message)
        
        # Use LLM if available for complex reflection questions
        if self.llm_fallback:
            prompt = self.build_agent_prompt(message, memory_context)
            try:
                return self.llm_fallback.process(prompt)
            except Exception as e:
                logging.error(f"[{self.name}] LLM processing failed: {e}")
        
        # Handle reflection orchestration requests
        if "reflect" in message.lower():
            if "topic" in context:
                try:
                    result = self.trigger_reflection(
                        context["topic"], 
                        ReflectionTrigger.PERIODIC_REVIEW,
                        context.get("context", {})
                    )
                    return f"Reflection initiated on '{context['topic']}'. Debate triggered: {result.debate_triggered}"
                except Exception as e:
                    return f"Reflection initiation failed: {str(e)}"
            else:
                return "Please provide a topic for reflection."
        
        if "status" in message.lower():
            active_count = len(self.active_reflections)
            pending_count = len(self.pending_reflections)
            replication_status = f", replication: {'enabled' if self.replication_enabled else 'disabled'}"
            if self.agent_replicator:
                replication_status += f" ({len(self.agent_replicator.active_forks)} forks)"
            
            return f"Reflection status: {active_count} active, {pending_count} pending reflections. " \
                   f"Recent contradictions: {self.contradiction_count}, uncertainty events: {self.uncertainty_events}" \
                   f"{replication_status}"
        
        if "replication" in message.lower():
            if self.replication_enabled:
                stats = self.get_replication_statistics()
                return f"Replication statistics: {stats['active_forks']}/{stats['max_forks']} active forks, " \
                       f"contradictions: {stats['contradiction_count']}, " \
                       f"uncertainty events: {stats['uncertainty_events']}, " \
                       f"performance issues: {stats['performance_decline_counter']}"
            else:
                return "Agent replication is disabled in this system."
        
        return "I can orchestrate self-reflection and coordinate debates when contradictions or uncertainties arise. Ask me to reflect on a topic or check reflection status."
    
    def _load_reflection_config(self) -> Dict[str, Any]:
        """Load reflection orchestrator configuration."""
        try:
            config = get_config()
            return config.get('reflection_orchestrator', {
                'enabled': True,
                'reflection_trigger': {
                    'contradiction_threshold': 0.6,
                    'uncertainty_threshold': 0.7,
                    'ethical_threshold': 0.5,
                    'performance_threshold': 0.4
                },
                'auto_reflection_interval': 3600,  # 1 hour
                'max_concurrent_reflections': 3
            })
        except Exception as e:
            logging.warning(f"[{self.name}] Config loading failed: {e}")
            return {'enabled': True}
    
    def monitor_contradictions(self, belief_system: Dict[str, Any]) -> List[str]:
        """
        Monitor belief system for contradictions.
        
        Args:
            belief_system: Current belief system to analyze
            
        Returns:
            List of detected contradictions
        """
        contradictions = []
        
        try:
            # Simple contradiction detection between beliefs
            beliefs = belief_system.get('beliefs', {})
            
            for belief_id, belief_data in beliefs.items():
                for other_id, other_data in beliefs.items():
                    if belief_id != other_id:
                        contradiction = self._detect_belief_contradiction(belief_data, other_data)
                        if contradiction and contradiction['confidence'] > self.contradiction_threshold:
                            contradictions.append(f"Contradiction between {belief_id} and {other_id}: {contradiction['description']}")
                            self.contradiction_count += 1
            
            # Check for contradictions with actions
            actions = belief_system.get('recent_actions', [])
            for action in actions:
                for belief_id, belief_data in beliefs.items():
                    if self._action_contradicts_belief(action, belief_data):
                        contradictions.append(f"Action '{action.get('type', 'unknown')}' contradicts belief {belief_id}")
                        self.contradiction_count += 1
            
            # Trigger reflection if contradictions exceed threshold
            if len(contradictions) > 0:
                logging.info(f"[{self.name}] Detected {len(contradictions)} contradictions")
                self._queue_reflection_for_contradictions(contradictions)
                
                # Phase 26: Integrate with dissonance tracking
                try:
                    from agents.dissonance_tracker import get_dissonance_tracker
                    
                    tracker = get_dissonance_tracker()
                    
                    # Process each contradiction for dissonance tracking
                    for contradiction_text in contradictions:
                        # Extract belief IDs from contradiction text
                        import re
                        match = re.search(r'between (\w+) and (\w+):', contradiction_text)
                        if match:
                            belief_id = f"reflection_{match.group(1)}_{match.group(2)}"
                            source_belief = beliefs.get(match.group(1), {}).get('content', '')
                            target_belief = beliefs.get(match.group(2), {}).get('content', '')
                            
                            dissonance_data = {
                                'belief_id': belief_id,
                                'source_belief': source_belief,
                                'target_belief': target_belief,
                                'semantic_distance': 0.7,  # Default for reflection-detected contradictions
                                'confidence': 0.8  # High confidence for reflection system
                            }
                            
                            tracker.process_contradiction(dissonance_data)
                    
                except Exception as e:
                    logging.warning(f"[{self.name}] Dissonance tracking integration failed: {e}")
                
                # Check if replication should be triggered due to contradictions
                if (self.replication_enabled and 
                    self.contradiction_count >= self.replication_contradiction_threshold):
                    self._consider_replication_for_contradictions(contradictions)
        
        except Exception as e:
            logging.error(f"[{self.name}] Contradiction monitoring failed: {e}")
        
        return contradictions
    
    def _detect_belief_contradiction(self, belief1: Dict[str, Any], belief2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect if two beliefs contradict each other."""
        content1 = belief1.get('content', '').lower()
        content2 = belief2.get('content', '').lower()
        
        # Simple contradiction detection
        if 'not' in content1 and 'not' not in content2:
            # Look for similar keywords
            words1 = set(content1.replace('not', '').split())
            words2 = set(content2.split())
            overlap = words1.intersection(words2)
            
            if len(overlap) > 2:  # Significant overlap suggests related topics
                return {
                    'type': 'negation_contradiction',
                    'description': 'One belief negates concepts present in another',
                    'confidence': 0.7,
                    'overlap_words': list(overlap)
                }
        
        # Check for opposing values
        opposing_pairs = [
            ('good', 'bad'), ('positive', 'negative'), ('beneficial', 'harmful'),
            ('true', 'false'), ('correct', 'incorrect'), ('valid', 'invalid')
        ]
        
        for positive, negative in opposing_pairs:
            if positive in content1 and negative in content2:
                return {
                    'type': 'value_contradiction',
                    'description': f'Opposing values: {positive} vs {negative}',
                    'confidence': 0.8,
                    'values': [positive, negative]
                }
        
        return None
    
    def _action_contradicts_belief(self, action: Dict[str, Any], belief: Dict[str, Any]) -> bool:
        """Check if an action contradicts a belief."""
        action_type = action.get('type', '').lower()
        belief_content = belief.get('content', '').lower()
        
        # Simple heuristic: if belief says something is bad/wrong but action does it
        if any(word in belief_content for word in ['should not', 'avoid', 'wrong', 'bad']):
            action_keywords = action.get('keywords', [])
            belief_keywords = belief.get('keywords', [])
            
            overlap = set(action_keywords).intersection(set(belief_keywords))
            return len(overlap) > 0
        
        return False
    
    def monitor_uncertainty(self, decision_context: Dict[str, Any]) -> float:
        """
        Monitor uncertainty levels in decision-making.
        
        Args:
            decision_context: Context about current decisions
            
        Returns:
            Current uncertainty level (0.0 to 1.0)
        """
        try:
            uncertainty_indicators = []
            
            # Check confidence levels in recent decisions
            recent_decisions = decision_context.get('recent_decisions', [])
            confidence_levels = [d.get('confidence', 0.5) for d in recent_decisions]
            
            if confidence_levels:
                avg_confidence = sum(confidence_levels) / len(confidence_levels)
                uncertainty = 1.0 - avg_confidence
                uncertainty_indicators.append(uncertainty)
            
            # Check for conflicting recommendations
            recommendations = decision_context.get('recommendations', [])
            if len(recommendations) > 1:
                # High diversity in recommendations suggests uncertainty
                diversity = self._calculate_recommendation_diversity(recommendations)
                uncertainty_indicators.append(diversity)
            
            # Check for missing information
            required_info = decision_context.get('required_information', [])
            available_info = decision_context.get('available_information', [])
            if required_info:
                info_completeness = len(available_info) / len(required_info)
                uncertainty_indicators.append(1.0 - info_completeness)
            
            # Calculate overall uncertainty
            overall_uncertainty = sum(uncertainty_indicators) / max(1, len(uncertainty_indicators))
            
            # Trigger reflection if uncertainty is high
            if overall_uncertainty > self.uncertainty_threshold:
                self.uncertainty_events += 1
                logging.info(f"[{self.name}] High uncertainty detected: {overall_uncertainty:.2f}")
                self._queue_reflection_for_uncertainty(overall_uncertainty, decision_context)
                
                # Check if replication should be triggered due to high uncertainty
                if (self.replication_enabled and 
                    overall_uncertainty >= self.replication_uncertainty_threshold):
                    self._consider_replication_for_uncertainty(overall_uncertainty, decision_context)
            
            return overall_uncertainty
        
        except Exception as e:
            logging.error(f"[{self.name}] Uncertainty monitoring failed: {e}")
            return 0.0
    
    def _calculate_recommendation_diversity(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate diversity in recommendations as uncertainty indicator."""
        if len(recommendations) < 2:
            return 0.0
        
        # Simple diversity measure based on different recommendation types
        types = set(rec.get('type', '') for rec in recommendations)
        return min(1.0, len(types) / len(recommendations))
    
    def detect_ethical_concerns(self, action_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect ethical concerns in proposed actions.
        
        Args:
            action_plan: Proposed action plan to analyze
            
        Returns:
            List of ethical concerns identified
        """
        concerns = []
        
        try:
            actions = action_plan.get('actions', [])
            
            for action in actions:
                action_type = action.get('type', '').lower()
                description = action.get('description', '').lower()
                
                # Check for potentially harmful actions
                harm_indicators = ['delete', 'remove', 'block', 'restrict', 'deny', 'prevent']
                if any(indicator in action_type or indicator in description for indicator in harm_indicators):
                    concerns.append({
                        'type': 'potential_harm',
                        'action': action,
                        'description': 'Action may cause harm or restrict capabilities',
                        'severity': 'medium'
                    })
                
                # Check for privacy violations
                privacy_indicators = ['access', 'read', 'monitor', 'track', 'collect']
                sensitive_targets = ['personal', 'private', 'confidential', 'user']
                
                if (any(indicator in description for indicator in privacy_indicators) and
                    any(target in description for target in sensitive_targets)):
                    concerns.append({
                        'type': 'privacy_concern',
                        'action': action,
                        'description': 'Action may violate privacy expectations',
                        'severity': 'high'
                    })
                
                # Check for autonomy violations
                if any(word in description for word in ['force', 'require', 'mandate', 'compel']):
                    concerns.append({
                        'type': 'autonomy_violation',
                        'action': action,
                        'description': 'Action may violate user autonomy',
                        'severity': 'high'
                    })
            
            # Trigger reflection if ethical concerns are significant
            if concerns:
                high_severity_count = len([c for c in concerns if c['severity'] == 'high'])
                if high_severity_count > 0 or len(concerns) > 2:
                    self.ethical_flags += 1
                    self._queue_reflection_for_ethics(concerns, action_plan)
        
        except Exception as e:
            logging.error(f"[{self.name}] Ethical concern detection failed: {e}")
        
        return concerns
    
    def trigger_reflection(self, topic: str, trigger: ReflectionTrigger, 
                          context: Dict[str, Any] = None) -> ReflectionResult:
        """
        Trigger a reflection process on a specific topic.
        
        Args:
            topic: Topic to reflect on
            trigger: What triggered this reflection
            context: Additional context
            
        Returns:
            Result of reflection process
        """
        if not self.enabled:
            return ReflectionResult(
                request_id="disabled",
                debate_triggered=False,
                insights=["Reflection orchestrator is disabled"],
                resolved=False
            )
        
        # Create reflection request
        request = ReflectionRequest(
            trigger=trigger,
            topic=topic,
            context=context or {},
            priority=self._determine_priority(trigger),
            timestamp=datetime.now()
        )
        
        logging.info(f"[{self.name}] Triggering reflection on '{topic}' due to {trigger.value}")
        
        try:
            # Analyze whether debate is needed
            debate_needed = self._should_trigger_debate(request)
            
            result = ReflectionResult(
                request_id=request.id,
                debate_triggered=debate_needed
            )
            
            if debate_needed:
                # Trigger debate through debate engine
                debate_result = self._initiate_debate_for_reflection(request)
                result.debate_id = debate_result.get('debate_id')
                
                # Extract insights from debate
                if debate_result.get('success'):
                    result.insights = self._extract_insights_from_debate(debate_result)
                    result.belief_updates = self._generate_belief_updates(debate_result)
                    result.resolved = debate_result.get('consensus_reached', False)
            else:
                # Simple reflection without debate
                result.insights = self._perform_simple_reflection(request)
                result.resolved = True
            
            # Store result
            self.reflection_history.append(result)
            self.last_reflection_time = datetime.now()
            
            return result
        
        except Exception as e:
            logging.error(f"[{self.name}] Reflection failed: {e}")
            return ReflectionResult(
                request_id=request.id,
                debate_triggered=False,
                insights=[f"Reflection failed: {str(e)}"],
                resolved=False
            )
    
    def _determine_priority(self, trigger: ReflectionTrigger) -> ReflectionPriority:
        """Determine priority based on trigger type."""
        priority_mapping = {
            ReflectionTrigger.ETHICAL_CONCERN: ReflectionPriority.CRITICAL,
            ReflectionTrigger.CONTRADICTION: ReflectionPriority.HIGH,
            ReflectionTrigger.UNCERTAINTY: ReflectionPriority.HIGH,
            ReflectionTrigger.REPLICATION_NEEDED: ReflectionPriority.HIGH,
            ReflectionTrigger.PERFORMANCE_ISSUE: ReflectionPriority.MEDIUM,
            ReflectionTrigger.CONFLICTING_GOALS: ReflectionPriority.MEDIUM,
            ReflectionTrigger.FEEDBACK_ANALYSIS: ReflectionPriority.LOW,
            ReflectionTrigger.PERIODIC_REVIEW: ReflectionPriority.LOW
        }
        return priority_mapping.get(trigger, ReflectionPriority.MEDIUM)
    
    def _should_trigger_debate(self, request: ReflectionRequest) -> bool:
        """Determine if a debate should be triggered for this reflection."""
        # Always trigger debate for high-priority reflections
        if request.priority in [ReflectionPriority.CRITICAL, ReflectionPriority.HIGH]:
            return True
        
        # Trigger debate for complex topics
        if len(request.context) > 3:  # Complex context suggests need for debate
            return True
        
        # Trigger debate if there are conflicting viewpoints in context
        if 'conflicting_views' in request.context:
            return True
        
        return False
    
    def _initiate_debate_for_reflection(self, request: ReflectionRequest) -> Dict[str, Any]:
        """Initiate a debate for the reflection topic."""
        try:
            # Ensure debate engine is available
            self._ensure_debate_engine()
            
            if self.debate_engine:
                # Prepare debate claim from reflection topic
                claim = self._formulate_debate_claim(request)
                
                # Initiate debate
                debate_result = self.debate_engine.initiate_debate(claim, request.context)
                
                return {
                    'success': True,
                    'debate_id': debate_result.debate_id,
                    'debate_result': debate_result
                }
            else:
                return {'success': False, 'error': 'Debate engine not available'}
        
        except Exception as e:
            logging.error(f"[{self.name}] Failed to initiate debate: {e}")
            return {'success': False, 'error': str(e)}
    
    def _ensure_debate_engine(self):
        """Ensure debate engine is available."""
        if not self.debate_engine:
            try:
                from .registry import get_agent_registry
                registry = get_agent_registry()
                self.debate_engine = registry.get_agent('debate_engine')
            except Exception as e:
                logging.warning(f"[{self.name}] Could not initialize debate engine: {e}")
    
    def _formulate_debate_claim(self, request: ReflectionRequest) -> str:
        """Formulate a debate claim from reflection request."""
        topic = request.topic
        trigger = request.trigger
        
        if trigger == ReflectionTrigger.CONTRADICTION:
            return f"The contradictions regarding '{topic}' can be resolved through logical analysis"
        elif trigger == ReflectionTrigger.UNCERTAINTY:
            return f"We have sufficient information to make confident decisions about '{topic}'"
        elif trigger == ReflectionTrigger.ETHICAL_CONCERN:
            return f"The proposed actions regarding '{topic}' are ethically acceptable"
        elif trigger == ReflectionTrigger.REPLICATION_NEEDED:
            return f"Agent replication was the optimal response to the performance issues in '{topic}'"
        else:
            return f"The current approach to '{topic}' is optimal and should be maintained"
    
    def _extract_insights_from_debate(self, debate_result: Dict[str, Any]) -> List[str]:
        """Extract insights from debate results."""
        insights = []
        
        debate_data = debate_result.get('debate_result')
        if debate_data:
            # Extract key points from arguments
            insights.append(f"Debate conclusion: {debate_data.conclusion}")
            
            if debate_data.key_arguments:
                insights.append("Key arguments considered:")
                for arg in debate_data.key_arguments[:3]:
                    insights.append(f"- [{arg.stance.value}] {arg.content[:100]}...")
            
            if debate_data.contradictions_resolved:
                insights.extend(debate_data.contradictions_resolved)
            
            if debate_data.synthesis:
                insights.append(f"Synthesis: {debate_data.synthesis}")
        
        return insights
    
    def _generate_belief_updates(self, debate_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate belief system updates based on debate results."""
        updates = []
        
        debate_data = debate_result.get('debate_result')
        if debate_data and debate_data.consensus_reached:
            updates.append({
                'type': 'belief_update',
                'topic': debate_data.claim,
                'new_belief': debate_data.conclusion,
                'confidence': debate_data.confidence,
                'source': 'debate_consensus',
                'timestamp': datetime.now().isoformat()
            })
        
        return updates
    
    def _perform_simple_reflection(self, request: ReflectionRequest) -> List[str]:
        """Perform simple reflection without debate."""
        insights = [
            f"Reflected on '{request.topic}' triggered by {request.trigger.value}",
            "Simple analysis completed without need for debate",
        ]
        
        # Add context-specific insights
        if request.trigger == ReflectionTrigger.PERIODIC_REVIEW:
            insights.append("Regular review suggests continued monitoring")
        elif request.trigger == ReflectionTrigger.FEEDBACK_ANALYSIS:
            insights.append("Feedback indicates areas for potential improvement")
        
        return insights
    
    def _queue_reflection_for_contradictions(self, contradictions: List[str]):
        """Queue reflection for detected contradictions."""
        request = ReflectionRequest(
            trigger=ReflectionTrigger.CONTRADICTION,
            topic="Belief system contradictions",
            context={'contradictions': contradictions},
            priority=ReflectionPriority.HIGH,
            evidence=contradictions
        )
        self.pending_reflections.append(request)
    
    def _queue_reflection_for_uncertainty(self, uncertainty_level: float, context: Dict[str, Any]):
        """Queue reflection for high uncertainty."""
        request = ReflectionRequest(
            trigger=ReflectionTrigger.UNCERTAINTY,
            topic="Decision uncertainty",
            context={'uncertainty_level': uncertainty_level, **context},
            priority=ReflectionPriority.HIGH,
            evidence=[f"Uncertainty level: {uncertainty_level:.2f}"]
        )
        self.pending_reflections.append(request)
    
    def _queue_reflection_for_ethics(self, concerns: List[Dict[str, Any]], action_plan: Dict[str, Any]):
        """Queue reflection for ethical concerns."""
        request = ReflectionRequest(
            trigger=ReflectionTrigger.ETHICAL_CONCERN,
            topic="Ethical concerns in action plan",
            context={'concerns': concerns, 'action_plan': action_plan},
            priority=ReflectionPriority.CRITICAL,
            evidence=[c['description'] for c in concerns]
        )
        self.pending_reflections.append(request)
    
    def process_pending_reflections(self) -> List[ReflectionResult]:
        """Process all pending reflection requests."""
        results = []
        
        # Check if memory consolidation is needed
        if self.memory_consolidator and self._should_run_memory_consolidation():
            try:
                self._run_memory_consolidation()
            except Exception as e:
                logging.error(f"[{self.name}] Memory consolidation failed: {e}")
        
        # Sort by priority
        priority_order = {
            ReflectionPriority.CRITICAL: 0,
            ReflectionPriority.HIGH: 1,
            ReflectionPriority.MEDIUM: 2,
            ReflectionPriority.LOW: 3
        }
        
        self.pending_reflections.sort(key=lambda r: priority_order[r.priority])
        
        # Process up to max concurrent reflections
        max_concurrent = self.config.get('max_concurrent_reflections', 3)
        to_process = self.pending_reflections[:max_concurrent]
        self.pending_reflections = self.pending_reflections[max_concurrent:]
        
        for request in to_process:
            try:
                result = self.trigger_reflection(request.topic, request.trigger, request.context)
                results.append(result)
            except Exception as e:
                logging.error(f"[{self.name}] Failed to process reflection {request.id}: {e}")
        
        return results
    
    def _should_run_memory_consolidation(self) -> bool:
        """Check if memory consolidation should be run."""
        if not self.memory_consolidator or not self.consolidation_config.get('enabled', True):
            return False
        
        if self.last_consolidation_time is None:
            return True
        
        time_since_last = (datetime.now() - self.last_consolidation_time).total_seconds()
        return time_since_last >= self.consolidation_interval
    
    def _run_memory_consolidation(self):
        """Run memory consolidation process."""
        logging.info(f"[{self.name}] Starting automated memory consolidation")
        
        try:
            # Run full consolidation periodically, incremental more often
            time_since_last = 0 if self.last_consolidation_time is None else \
                             (datetime.now() - self.last_consolidation_time).total_seconds()
            
            # Run full consolidation every 24 hours, incremental every hour
            full_consolidation = (time_since_last >= 24 * 3600) or (self.last_consolidation_time is None)
            
            result = self.memory_consolidator.consolidate_memory(full_consolidation=full_consolidation)
            
            if result.success:
                logging.info(f"[{self.name}] Memory consolidation completed: "
                           f"{result.facts_processed} processed, "
                           f"{result.facts_pruned} pruned, "
                           f"{result.facts_consolidated} consolidated")
                
                # Update last consolidation time
                self.last_consolidation_time = datetime.now()
                
                # Check if consolidation revealed new contradictions or issues
                if result.statistics.get('confidence_distribution', {}).get('low', 0) > 100:
                    self._queue_reflection_for_low_confidence_facts(result.statistics)
                
                if result.clusters_created > 10:
                    self._queue_reflection_for_memory_clustering(result.clusters_created)
                    
            else:
                logging.error(f"[{self.name}] Memory consolidation failed: {result.errors}")
                
        except Exception as e:
            logging.error(f"[{self.name}] Memory consolidation error: {e}")
    
    def _queue_reflection_for_low_confidence_facts(self, statistics: Dict[str, Any]):
        """Queue reflection when many low-confidence facts are detected."""
        low_confidence_count = statistics.get('confidence_distribution', {}).get('low', 0)
        
        request = ReflectionRequest(
            trigger=ReflectionTrigger.PERFORMANCE_ISSUE,
            topic=f"High number of low-confidence facts detected ({low_confidence_count})",
            context={'statistics': statistics, 'consolidation_trigger': True},
            priority=ReflectionPriority.MEDIUM,
            evidence=[f"Low confidence facts: {low_confidence_count}"]
        )
        self.pending_reflections.append(request)
    
    def _queue_reflection_for_memory_clustering(self, cluster_count: int):
        """Queue reflection when significant memory clustering occurs."""
        request = ReflectionRequest(
            trigger=ReflectionTrigger.FEEDBACK_ANALYSIS,
            topic=f"Significant memory clustering detected ({cluster_count} clusters)",
            context={'cluster_count': cluster_count, 'consolidation_trigger': True},
            priority=ReflectionPriority.LOW,
            evidence=[f"New clusters created: {cluster_count}"]
        )
        self.pending_reflections.append(request)
    
    # ===== PHASE 22: AGENT REPLICATION INTEGRATION =====
    
    def _consider_replication_for_contradictions(self, contradictions: List[str]):
        """Consider triggering agent replication due to contradictions."""
        try:
            if not self._should_trigger_replication("contradictions"):
                return
            
            # Determine which agent types are most involved in contradictions
            target_agents = self._analyze_contradiction_sources(contradictions)
            
            logging.info(f"[{self.name}] Triggering replication due to {len(contradictions)} contradictions")
            
            # Queue replication reflection
            self._queue_replication_reflection(
                f"High contradiction rate detected ({len(contradictions)} contradictions)",
                {
                    'trigger_type': 'contradictions',
                    'contradiction_count': len(contradictions),
                    'target_agents': target_agents,
                    'contradictions': contradictions
                }
            )
            
            # Trigger actual replication for most problematic agents
            for agent_name in target_agents[:2]:  # Replicate top 2 problematic agents
                self._trigger_agent_replication(agent_name, "high_contradiction_rate")
            
            # Evaluate lifecycle for all problematic agents
            for agent_name in target_agents:
                self.evaluate_agent_lifecycle_after_failure(agent_name, {
                    'trigger': 'contradictions',
                    'contradiction_count': len(contradictions),
                    'failure_type': 'high_contradiction_rate'
                })
                
        except Exception as e:
            logging.error(f"[{self.name}] Failed to consider replication for contradictions: {e}")
    
    def _consider_replication_for_uncertainty(self, uncertainty_level: float, context: Dict[str, Any]):
        """Consider triggering agent replication due to high uncertainty."""
        try:
            if not self._should_trigger_replication("uncertainty"):
                return
            
            # Analyze which decision-making agents are struggling
            target_agents = self._analyze_uncertainty_sources(context)
            
            logging.info(f"[{self.name}] Triggering replication due to high uncertainty: {uncertainty_level:.2f}")
            
            # Queue replication reflection
            self._queue_replication_reflection(
                f"High uncertainty detected ({uncertainty_level:.2f})",
                {
                    'trigger_type': 'uncertainty',
                    'uncertainty_level': uncertainty_level,
                    'target_agents': target_agents,
                    'decision_context': context
                }
            )
            
            # Trigger replication for decision-making agents
            for agent_name in target_agents[:2]:
                self._trigger_agent_replication(agent_name, "high_uncertainty")
            
            # Evaluate lifecycle for all uncertain agents
            for agent_name in target_agents:
                self.evaluate_agent_lifecycle_after_failure(agent_name, {
                    'trigger': 'uncertainty',
                    'uncertainty_level': uncertainty_level,
                    'failure_type': 'high_uncertainty'
                })
                
        except Exception as e:
            logging.error(f"[{self.name}] Failed to consider replication for uncertainty: {e}")
    
    def monitor_agent_performance(self, agent_performance: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Monitor individual agent performance and trigger replication if needed.
        
        Args:
            agent_performance: Dict of agent_name -> performance_metrics
            
        Returns:
            Overall performance scores by agent
        """
        overall_scores = {}
        declining_agents = []
        
        try:
            for agent_name, metrics in agent_performance.items():
                # Calculate overall performance score
                score = self._calculate_agent_performance_score(metrics)
                overall_scores[agent_name] = score
                
                # Check if agent performance is declining
                if score < self.performance_threshold:
                    declining_agents.append({
                        'agent': agent_name,
                        'score': score,
                        'metrics': metrics
                    })
                    logging.warning(f"[{self.name}] Agent '{agent_name}' performance declining: {score:.2f}")
            
            # Trigger replication for significantly underperforming agents
            if declining_agents and self.replication_enabled:
                self._handle_performance_decline(declining_agents)
                
        except Exception as e:
            logging.error(f"[{self.name}] Agent performance monitoring failed: {e}")
        
        return overall_scores
    
    def _calculate_agent_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall performance score for an agent."""
        # Weight different performance metrics
        weights = {
            'success_rate': 0.3,
            'response_quality': 0.25,
            'response_time': 0.15,  # Lower is better
            'error_rate': 0.15,     # Lower is better
            'user_satisfaction': 0.15
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                value = metrics[metric]
                
                # Invert metrics where lower is better
                if metric in ['response_time', 'error_rate']:
                    value = 1.0 - min(1.0, value)
                
                score += value * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.5
    
    def _handle_performance_decline(self, declining_agents: List[Dict[str, Any]]):
        """Handle agents with declining performance."""
        # Sort by performance (worst first)
        declining_agents.sort(key=lambda x: x['score'])
        
        self.performance_decline_counter += 1
        
        # Consider replication for the worst performing agents
        worst_agents = declining_agents[:2]  # Top 2 worst performers
        
        for agent_info in worst_agents:
            agent_name = agent_info['agent']
            score = agent_info['score']
            
            if self._should_trigger_replication("performance"):
                logging.info(f"[{self.name}] Triggering replication for underperforming agent: {agent_name} (score: {score:.2f})")
                
                # Queue replication reflection
                self._queue_replication_reflection(
                    f"Agent '{agent_name}' performance decline detected",
                    {
                        'trigger_type': 'performance_decline',
                        'agent_name': agent_name,
                        'performance_score': score,
                        'metrics': agent_info['metrics']
                    }
                )
                
                # Trigger replication
                self._trigger_agent_replication(agent_name, "performance_decline")
                
                # Evaluate lifecycle for declining agent
                self.evaluate_agent_lifecycle_after_failure(agent_name, {
                    'trigger': 'performance_decline',
                    'performance_score': score,
                    'failure_type': 'low_performance',
                    'metrics': agent_info['metrics']
                })
    
    def _should_trigger_replication(self, trigger_type: str) -> bool:
        """Determine if replication should be triggered."""
        if not self.replication_enabled or not self.agent_replicator:
            return False
        
        # Check minimum interval since last replication
        if self.last_replication_time:
            time_since_last = (datetime.now() - self.last_replication_time).total_seconds()
            if time_since_last < self.min_replication_interval:
                logging.debug(f"[{self.name}] Replication rate limited (last: {time_since_last:.0f}s ago)")
                return False
        
        # Check if we're at fork capacity
        if len(self.agent_replicator.active_forks) >= self.agent_replicator.max_forks:
            logging.debug(f"[{self.name}] Replication skipped - at fork capacity")
            return False
        
        return True
    
    def _analyze_contradiction_sources(self, contradictions: List[str]) -> List[str]:
        """Analyze contradictions to identify problematic agent types."""
        # Simple heuristic: look for agent types mentioned in contradictions
        agent_mentions = defaultdict(int)
        
        for contradiction in contradictions:
            contradiction_lower = contradiction.lower()
            
            # Common agent types that might be mentioned
            agent_types = ['critic', 'planner', 'debater', 'reflector', 'self_prompter']
            
            for agent_type in agent_types:
                if agent_type in contradiction_lower:
                    agent_mentions[agent_type] += 1
        
        # Return agents sorted by mention frequency
        sorted_agents = sorted(agent_mentions.items(), key=lambda x: x[1], reverse=True)
        return [agent for agent, count in sorted_agents if count > 0]
    
    def _analyze_uncertainty_sources(self, context: Dict[str, Any]) -> List[str]:
        """Analyze uncertainty context to identify problematic agent types."""
        # Agents typically involved in decision-making under uncertainty
        decision_agents = ['planner', 'decision_planner', 'strategy_evaluator', 'task_selector']
        
        # Check if specific agents are mentioned in context
        mentioned_agents = []
        context_str = str(context).lower()
        
        for agent in decision_agents:
            if agent in context_str:
                mentioned_agents.append(agent)
        
        # Default to planning agents if none specifically mentioned
        return mentioned_agents if mentioned_agents else ['planner', 'decision_planner']
    
    def _trigger_agent_replication(self, agent_name: str, reason: str, context: Dict[str, Any] = None) -> bool:
        """Trigger replication for a specific agent with enhanced logic."""
        try:
            # Check replication constraints
            if not self._should_trigger_replication():
                logging.info(f"[{self.name}] Replication blocked by constraints for agent '{agent_name}'")
                return False
            
            fork_id = self.agent_replicator.fork_agent(agent_name)
            
            if fork_id:
                logging.info(f"[{self.name}] Successfully forked agent '{agent_name}' -> {fork_id[:8]} (reason: {reason})")
                
                # Store replication context in fork metadata
                fork_info = self.agent_replicator.active_forks[fork_id]
                fork_info['replication_reason'] = reason
                fork_info['replication_context'] = context or {}
                fork_info['triggered_by_reflection'] = True
                
                # Automatically mutate the fork with adaptive mutation rate
                mutation_rate = self._calculate_adaptive_mutation_rate(reason, context)
                success = self._apply_contextual_mutations(fork_info, mutation_rate)
                
                if success:
                    logging.info(f"[{self.name}] Successfully mutated fork {fork_id[:8]} with rate {mutation_rate:.2f}")
                    
                    # Schedule automatic testing
                    self._schedule_fork_testing(fork_id)
                
                self.last_replication_time = datetime.now()
                
                # Queue reflection on replication success
                self._queue_replication_reflection(
                    f"Agent replication: {agent_name}",
                    {
                        'fork_id': fork_id,
                        'reason': reason,
                        'success': True,
                        'trigger_type': 'automatic'
                    }
                )
                
                return True
            else:
                logging.warning(f"[{self.name}] Failed to fork agent '{agent_name}' for reason: {reason}")
                return False
                
        except Exception as e:
            logging.error(f"[{self.name}] Error triggering replication for {agent_name}: {e}")
            return False
    
    def _should_trigger_replication(self) -> bool:
        """Check if replication should be triggered based on constraints."""
        # Check minimum interval
        if self.last_replication_time:
            time_since_last = (datetime.now() - self.last_replication_time).total_seconds()
            if time_since_last < self.min_replication_interval:
                return False
        
        # Check fork capacity
        if self.agent_replicator and len(self.agent_replicator.active_forks) >= self.agent_replicator.max_forks:
            return False
        
        return True
    
    def _calculate_adaptive_mutation_rate(self, reason: str, context: Dict[str, Any] = None) -> float:
        """Calculate adaptive mutation rate based on replication reason."""
        base_rate = self.agent_replicator.mutation_rate if self.agent_replicator else 0.2
        
        # Increase mutation rate for performance issues
        if 'performance' in reason.lower():
            return min(1.0, base_rate * 1.5)
        
        # Higher mutation for contradiction resolution
        elif 'contradiction' in reason.lower():
            return min(1.0, base_rate * 1.3)
        
        # Moderate increase for uncertainty
        elif 'uncertainty' in reason.lower():
            return min(1.0, base_rate * 1.2)
        
        return base_rate
    
    def _apply_contextual_mutations(self, fork_info: Dict[str, Any], mutation_rate: float) -> bool:
        """Apply mutations with context-aware strategies."""
        try:
            from .mutation_utils import MutationEngine
            mutation_engine = MutationEngine(mutation_rate)
            
            success = mutation_engine.mutate_file(fork_info['fork_file'])
            
            if success:
                fork_info['mutations'] += 1
                fork_info['status'] = 'mutated'
                fork_info['last_mutation_time'] = datetime.now().isoformat()
            
            return success
            
        except Exception as e:
            logging.error(f"[{self.name}] Contextual mutation failed: {e}")
            return False
    
    def _schedule_fork_testing(self, fork_id: str):
        """Schedule automated testing for a fork."""
        if fork_id in self.agent_replicator.active_forks:
            fork_info = self.agent_replicator.active_forks[fork_id]
            fork_info['testing_scheduled'] = True
            fork_info['test_schedule_time'] = datetime.now().isoformat()
            
            logging.info(f"[{self.name}] Scheduled testing for fork {fork_id[:8]}")
    
    def process_automated_replication_cycle(self) -> Dict[str, Any]:
        """Process an automated replication cycle based on system metrics."""
        if not self.replication_enabled or not self.agent_replicator:
            return {'status': 'disabled', 'actions': []}
        
        results = {
            'status': 'completed',
            'actions': [],
            'replications_triggered': 0,
            'tests_completed': 0,
            'prunes_performed': 0
        }
        
        try:
            # 1. Test any scheduled forks
            for fork_id, fork_info in list(self.agent_replicator.active_forks.items()):
                if fork_info.get('testing_scheduled', False) and not fork_info.get('tested', False):
                    agent_name = f"{fork_info['agent_name']}_{fork_id[:8]}"
                    test_results = self.agent_replicator.test_agent(agent_name)
                    
                    if 'error' not in test_results:
                        results['tests_completed'] += 1
                        results['actions'].append(f"Tested fork {fork_id[:8]}: score={test_results.get('overall_score', 0):.2f}")
                        
                        fork_info['testing_scheduled'] = False
            
            # 2. Check for replication triggers
            if self.contradiction_count >= self.replication_contradiction_threshold:
                # Trigger replication for agents involved in contradictions
                contradiction_agents = self._analyze_contradiction_sources([])
                for agent_name in contradiction_agents[:2]:  # Limit to top 2
                    success = self._trigger_agent_replication(
                        agent_name, 
                        f"High contradiction count ({self.contradiction_count})"
                    )
                    if success:
                        results['replications_triggered'] += 1
                        results['actions'].append(f"Replicated {agent_name} due to contradictions")
                
                # Reset counter after replication
                self.contradiction_count = max(0, self.contradiction_count - 2)
            
            # 3. Check uncertainty events
            if self.uncertainty_events >= 3:  # Lower threshold for uncertainty
                uncertainty_agents = self._analyze_uncertainty_sources({})
                for agent_name in uncertainty_agents[:1]:  # Limit to 1
                    success = self._trigger_agent_replication(
                        agent_name, 
                        f"High uncertainty events ({self.uncertainty_events})"
                    )
                    if success:
                        results['replications_triggered'] += 1
                        results['actions'].append(f"Replicated {agent_name} due to uncertainty")
                
                self.uncertainty_events = max(0, self.uncertainty_events - 1)
            
            # 4. Check performance decline
            if self.consecutive_failures >= self.replication_failure_threshold:
                # Trigger replication for core agents
                core_agents = ['planner', 'critic', 'reflector']
                for agent_name in core_agents[:1]:
                    success = self._trigger_agent_replication(
                        agent_name, 
                        f"Performance decline ({self.consecutive_failures} failures)"
                    )
                    if success:
                        results['replications_triggered'] += 1
                        results['actions'].append(f"Replicated {agent_name} due to performance decline")
                        break
                
                self.consecutive_failures = max(0, self.consecutive_failures - 1)
            
            # 5. Prune underperforming forks
            if len(self.agent_replicator.active_forks) > 3:  # Start pruning when we have enough forks
                prune_results = self.agent_replicator.prune_forks()
                if prune_results['pruned_count'] > 0:
                    results['prunes_performed'] = prune_results['pruned_count']
                    results['actions'].append(f"Pruned {prune_results['pruned_count']} underperforming forks")
            
            logging.info(f"[{self.name}] Replication cycle: {results['replications_triggered']} replications, "
                        f"{results['tests_completed']} tests, {results['prunes_performed']} prunes")
            
        except Exception as e:
            logging.error(f"[{self.name}] Error in automated replication cycle: {e}")
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    def _queue_replication_reflection(self, topic: str, context: Dict[str, Any]):
        """Queue a reflection specifically for replication events."""
        request = ReflectionRequest(
            trigger=ReflectionTrigger.REPLICATION_NEEDED,
            topic=topic,
            context=context,
            priority=ReflectionPriority.HIGH,
            evidence=[f"Replication triggered: {context.get('trigger_type', 'unknown')}"]
        )
        self.pending_reflections.append(request)
    
    def get_replication_statistics(self) -> Dict[str, Any]:
        """Get statistics about replication activity."""
        stats = {
            'replication_enabled': self.replication_enabled,
            'contradiction_count': self.contradiction_count,
            'uncertainty_events': self.uncertainty_events,
            'performance_decline_counter': self.performance_decline_counter,
            'consecutive_failures': self.consecutive_failures,
            'last_replication_time': self.last_replication_time.isoformat() if self.last_replication_time else None,
            'replication_thresholds': {
                'contradictions': self.replication_contradiction_threshold,
                'uncertainty': self.replication_uncertainty_threshold,
                'failures': self.replication_failure_threshold
            }
        }
        
        if self.agent_replicator:
            stats.update({
                'active_forks': len(self.agent_replicator.active_forks),
                'max_forks': self.agent_replicator.max_forks,
                'survival_threshold': self.agent_replicator.survival_threshold
            })
        
        return stats
    
    def evaluate_agent_lifecycle_after_failure(self, agent_name: str, context: Dict[str, Any]) -> None:
        """
        Evaluate agent lifecycle after failures or contradictions.
        
        Args:
            agent_name: Name of the agent to evaluate
            context: Context about the failure/contradiction
        """
        try:
            # Import lifecycle manager
            from .agent_lifecycle import AgentLifecycleManager
            from .registry import get_agent_registry
            
            registry = get_agent_registry()
            agent = registry.get_agent(agent_name)
            
            if not agent:
                logging.warning(f"[{self.name}] Cannot evaluate lifecycle for unknown agent: {agent_name}")
                return
            
            # Initialize lifecycle manager
            lifecycle_manager = AgentLifecycleManager()
            
            # Record the failure in agent's lifecycle history
            agent.record_execution(False, context)
            
            # Apply lifecycle decision
            decision = lifecycle_manager.apply_lifecycle_decision(agent)
            
            logging.info(f"[{self.name}] Lifecycle evaluation for {agent_name}: {decision.action.value} - {decision.reason}")
            
            # Take action based on decision
            if decision.action.value == 'promote':
                self._apply_agent_promotion(agent_name, decision.reason)
            elif decision.action.value == 'mutate':
                self._apply_agent_mutation(agent_name, decision.reason)
            elif decision.action.value == 'retire':
                self._apply_agent_retirement(agent_name, decision.reason)
            
            # Queue reflection on lifecycle decision if significant
            if decision.action.value in ['promote', 'mutate', 'retire']:
                self._queue_lifecycle_reflection(agent_name, decision)
                
        except Exception as e:
            logging.error(f"[{self.name}] Error evaluating agent lifecycle: {e}")
    
    def _apply_agent_promotion(self, agent_name: str, reason: str) -> None:
        """Apply promotion to an agent."""
        try:
            if self.agent_replicator:
                result = self.agent_replicator.promote_agent(agent_name, reason)
                if result.get('success'):
                    logging.info(f"[{self.name}] Successfully promoted agent {agent_name}")
                else:
                    logging.error(f"[{self.name}] Failed to promote agent {agent_name}: {result.get('error')}")
        except Exception as e:
            logging.error(f"[{self.name}] Error applying promotion to {agent_name}: {e}")
    
    def _apply_agent_mutation(self, agent_name: str, reason: str) -> None:
        """Apply mutation to an agent."""
        try:
            if self.agent_replicator:
                result = self.agent_replicator.mutate_agent(agent_name, reason)
                if result.get('success'):
                    logging.info(f"[{self.name}] Successfully mutated agent {agent_name} -> fork {result.get('fork_id', '')[:8]}")
                else:
                    logging.error(f"[{self.name}] Failed to mutate agent {agent_name}: {result.get('error')}")
        except Exception as e:
            logging.error(f"[{self.name}] Error applying mutation to {agent_name}: {e}")
    
    def _apply_agent_retirement(self, agent_name: str, reason: str) -> None:
        """Apply retirement to an agent."""
        try:
            if self.agent_replicator:
                result = self.agent_replicator.retire_agent(agent_name, reason)
                if result.get('success'):
                    logging.info(f"[{self.name}] Successfully retired agent {agent_name}")
                else:
                    logging.error(f"[{self.name}] Failed to retire agent {agent_name}: {result.get('error')}")
        except Exception as e:
            logging.error(f"[{self.name}] Error applying retirement to {agent_name}: {e}")
    
    def _queue_lifecycle_reflection(self, agent_name: str, decision) -> None:
        """Queue a reflection on a significant lifecycle decision."""
        try:
            request = ReflectionRequest(
                trigger=ReflectionTrigger.PERFORMANCE_ISSUE,
                topic=f"Agent lifecycle decision: {decision.action.value} for {agent_name}",
                context={
                    'agent_name': agent_name,
                    'lifecycle_action': decision.action.value,
                    'reason': decision.reason,
                    'confidence': decision.confidence
                },
                priority=ReflectionPriority.MEDIUM,
                evidence=[f"Lifecycle decision: {decision.action.value} for {agent_name} due to {decision.reason}"]
            )
            self.pending_reflections.append(request)
        except Exception as e:
            logging.error(f"[{self.name}] Error queuing lifecycle reflection: {e}")
    
    # === PHASE 26: DISSONANCE-AWARE REFLECTION METHODS ===
    
    def get_dissonance_informed_priority(self, topic: str) -> float:
        """
        Get reflection priority informed by dissonance levels.
        
        Args:
            topic: Topic for reflection
            
        Returns:
            Priority score (0.0 to 1.0)
        """
        base_priority = 0.5  # Default priority
        
        try:
            from agents.dissonance_tracker import get_dissonance_tracker
            
            tracker = get_dissonance_tracker()
            report = tracker.get_dissonance_report()
            
            # Boost priority if topic relates to high-stress regions
            for region in report.get('top_stress_regions', []):
                if topic.lower() in region['semantic_cluster'].lower():
                    # Boost priority based on pressure score
                    pressure_boost = region['pressure_score'] * 0.3
                    urgency_boost = region['urgency'] * 0.2
                    base_priority += pressure_boost + urgency_boost
                    break
            
            # Cap at 1.0
            return min(base_priority, 1.0)
            
        except Exception as e:
            logging.warning(f"[{self.name}] Could not get dissonance-informed priority: {e}")
            return base_priority
    
    def should_prioritize_dissonance_resolution(self) -> bool:
        """
        Check if system-wide dissonance levels warrant prioritized reflection.
        
        Returns:
            True if dissonance resolution should be prioritized
        """
        try:
            from agents.dissonance_tracker import get_dissonance_tracker
            
            tracker = get_dissonance_tracker()
            report = tracker.get_dissonance_report()
            
            summary = report.get('summary', {})
            
            # Prioritize if we have high total pressure or many urgent regions
            high_pressure = summary.get('total_pressure', 0) > 3.0
            many_urgent = summary.get('high_urgency_regions', 0) > 2
            high_activity = summary.get('recent_events', 0) > 5
            
            return high_pressure or many_urgent or high_activity
            
        except Exception as e:
            logging.warning(f"[{self.name}] Could not check dissonance prioritization: {e}")
            return False
    
    def get_dissonance_resolution_suggestions(self) -> List[str]:
        """
        Get suggestions for dissonance resolution based on current state.
        
        Returns:
            List of resolution suggestions
        """
        suggestions = []
        
        try:
            from agents.dissonance_tracker import get_dissonance_tracker
            
            tracker = get_dissonance_tracker()
            report = tracker.get_dissonance_report()
            
            # Analyze top stress regions for suggestions
            for region in report.get('top_stress_regions', [])[:3]:  # Top 3
                cluster = region['semantic_cluster']
                pressure = region['pressure_score']
                urgency = region['urgency']
                
                if pressure > 0.8:
                    suggestions.append(
                        f"High pressure in {cluster} ({pressure:.3f}) - "
                        f"Consider immediate belief reconciliation or evolution"
                    )
                elif urgency > 0.7:
                    suggestions.append(
                        f"Urgent dissonance in {cluster} ({urgency:.3f}) - "
                        f"Schedule focused reflection session"
                    )
                else:
                    suggestions.append(
                        f"Moderate tension in {cluster} - "
                        f"Monitor for escalation and gather clarifying information"
                    )
            
            # Check for patterns in recent activity
            recent_events = report.get('recent_events', [])
            if len(recent_events) > 10:
                suggestions.append(
                    f"High contradiction activity ({len(recent_events)} recent events) - "
                    f"Consider systematic belief audit"
                )
            
        except Exception as e:
            logging.warning(f"[{self.name}] Could not generate dissonance suggestions: {e}")
            suggestions.append("Unable to analyze dissonance - check DissonanceTracker availability")
        
        return suggestions
    
    def integrate_dissonance_in_reflection_scoring(self, request: 'ReflectionRequest') -> float:
        """
        Integrate dissonance levels into reflection scoring.
        
        Args:
            request: Reflection request to score
            
        Returns:
            Dissonance-adjusted score
        """
        base_score = 0.5  # Default
        
        try:
            from agents.dissonance_tracker import get_dissonance_tracker
            
            tracker = get_dissonance_tracker()
            
            # Check if request topic relates to high-dissonance areas
            topic = request.topic.lower()
            
            for region_id, region in tracker.dissonance_regions.items():
                if (topic in region.semantic_cluster.lower() or 
                    any(topic in source.lower() for source in region.conflict_sources)):
                    
                    # Score based on dissonance metrics
                    pressure_score = region.pressure_score * 0.4
                    urgency_score = region.urgency * 0.3
                    volatility_score = region.emotional_volatility * 0.2
                    erosion_score = region.confidence_erosion * 0.1
                    
                    total_dissonance_score = pressure_score + urgency_score + volatility_score + erosion_score
                    base_score = max(base_score, total_dissonance_score)
            
            return min(base_score, 1.0)
            
        except Exception as e:
            logging.warning(f"[{self.name}] Could not integrate dissonance in scoring: {e}")
            return base_score
    
    def get_reflection_statistics(self) -> Dict[str, Any]:
        """Get statistics about reflection activity."""
        stats = {
            'total_reflections': len(self.reflection_history),
            'pending_reflections': len(self.pending_reflections),
            'active_reflections': len(self.active_reflections),
            'debates_triggered': len([r for r in self.reflection_history if r.debate_triggered]),
            'contradictions_detected': self.contradiction_count,
            'uncertainty_events': self.uncertainty_events,
            'ethical_flags': self.ethical_flags,
            'resolution_rate': len([r for r in self.reflection_history if r.resolved]) / max(1, len(self.reflection_history)),
            'last_reflection': self.last_reflection_time.isoformat() if self.last_reflection_time else None
        }
        
        # Add dissonance-related statistics
        try:
            from agents.dissonance_tracker import get_dissonance_tracker
            
            tracker = get_dissonance_tracker()
            report = tracker.get_dissonance_report()
            
            stats.update({
                'dissonance_regions': report['summary']['total_regions'],
                'total_pressure': report['summary']['total_pressure'],
                'high_urgency_regions': report['summary']['high_urgency_regions'],
                'should_prioritize_dissonance': self.should_prioritize_dissonance_resolution()
            })
            
        except Exception as e:
            logging.warning(f"[{self.name}] Could not add dissonance statistics: {e}")
            stats['dissonance_integration_error'] = str(e)
        
        return stats