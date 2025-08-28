#!/usr/bin/env python3
"""
🎛️ Autonomous Memory Tuning System for MeRNSTA Phase 2

Auto-adjusts system parameters based on performance metrics:
- contradiction_threshold
- volatility_decay  
- belief_reinforcement_rate
- semantic_similarity_threshold

Monitors metrics like contradiction resolution rate, belief stability, 
clarification success, and automatically optimizes system performance.
"""

import logging
import time
import json
import math
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field

from .enhanced_memory_model import EnhancedTripletFact


@dataclass
class PerformanceMetrics:
    """Snapshot of system performance at a given time."""
    timestamp: float
    
    # Contradiction metrics
    contradiction_detection_rate: float  # Facts flagged as contradictory / total facts
    contradiction_resolution_rate: float  # Resolved contradictions / total contradictions
    false_positive_rate: float  # Incorrectly flagged contradictions / total flagged
    
    # Volatility metrics
    volatility_stability_ratio: float  # Stable beliefs / volatile beliefs
    volatility_reduction_rate: float  # Volatility reductions over time
    
    # Belief consolidation metrics
    consolidation_success_rate: float  # Successfully consolidated beliefs / attempts
    belief_reinforcement_effectiveness: float  # Strength of reinforced beliefs
    
    # User interaction metrics
    clarification_success_rate: float  # Successful clarifications / total requests
    user_satisfaction_proxy: float  # Based on engagement and resolution
    
    # System efficiency
    memory_utilization: float  # Active facts / total facts
    query_response_accuracy: float  # Relevant responses / total queries
    
    # Current parameter values (for tracking changes)
    parameters: Dict[str, float] = field(default_factory=dict)


@dataclass
class TuningAction:
    """Represents a parameter adjustment action."""
    timestamp: float
    parameter_name: str
    old_value: float
    new_value: float
    reason: str
    expected_improvement: str
    confidence: float  # How confident we are this will help (0.0-1.0)


class AutonomousMemoryTuning:
    """
    Continuously monitors system performance and automatically adjusts parameters
    to optimize contradiction detection, belief stability, and user satisfaction.
    """
    
    def __init__(self):
        from config.settings import DEFAULT_VALUES, SEMANTIC_SIMILARITY_THRESHOLD
        
        # Current system parameters - load from configuration
        self.parameters = {
            'contradiction_threshold': DEFAULT_VALUES.get('contradiction_threshold', 0.7),
            'volatility_decay_rate': 0.95,
            'belief_reinforcement_rate': 1.2,
            'semantic_similarity_threshold': SEMANTIC_SIMILARITY_THRESHOLD,
            'consolidation_threshold': DEFAULT_VALUES.get('consolidation_threshold', 0.85),
            'confabulation_threshold': DEFAULT_VALUES.get('confabulation_threshold', 0.7),
            'clarification_volatility_threshold': DEFAULT_VALUES.get('clarification_volatility_threshold', 0.7),
            'max_contradiction_age_days': 30.0
        }
        
        # Performance tracking
        self.metrics_history: deque = deque(maxlen=100)  # Keep last 100 snapshots
        self.tuning_history: List[TuningAction] = []
        self.last_tuning_time = 0.0
        
        # Tuning configuration
        self.tuning_interval = 3600  # 1 hour between tuning cycles
        self.min_data_points = 10   # Minimum metrics before tuning
        self.learning_rate = 0.1    # How aggressively to adjust parameters
        self.stability_window = 5   # Number of metrics to consider for trend analysis
        
        # Parameter bounds
        self.parameter_bounds = {
            'contradiction_threshold': (0.3, 0.9),
            'volatility_decay_rate': (0.8, 0.99),
            'belief_reinforcement_rate': (1.0, 2.0),
            'semantic_similarity_threshold': (0.2, 0.8),
            'consolidation_threshold': (0.6, 0.95),
            'confabulation_threshold': (0.4, 0.9),
            'clarification_volatility_threshold': (0.5, 0.9),
            'max_contradiction_age_days': (7.0, 90.0)
        }
        
        print("[AutonomousTuning] Initialized with default parameters")
    
    def collect_performance_metrics(self, facts: List[EnhancedTripletFact],
                                  recent_clarifications: List = None,
                                  recent_queries: List = None) -> PerformanceMetrics:
        """
        Collect current performance metrics from the system state.
        """
        current_time = time.time()
        
        # Calculate contradiction metrics
        total_facts = len(facts)
        contradictory_facts = [f for f in facts if f.contradiction]
        
        contradiction_detection_rate = len(contradictory_facts) / max(1, total_facts)
        
        # Estimate resolution rate based on recent contradiction activity
        recent_cutoff = current_time - (7 * 24 * 3600)  # Last 7 days
        recent_contradictions = [f for f in contradictory_facts 
                               if f.timestamp and f.timestamp > recent_cutoff]
        old_contradictions = [f for f in contradictory_facts 
                            if f.timestamp and f.timestamp <= recent_cutoff]
        
        # Resolution rate = old contradictions that are no longer active
        resolved_count = len([f for f in old_contradictions if not f.active])
        contradiction_resolution_rate = resolved_count / max(1, len(old_contradictions))
        
        # Estimate false positive rate (contradictions that were quickly resolved)
        quick_resolutions = [f for f in recent_contradictions 
                           if not f.active and (current_time - f.timestamp) < 3600]  # Resolved within 1 hour
        false_positive_rate = len(quick_resolutions) / max(1, len(recent_contradictions))
        
        # Calculate volatility metrics
        volatile_facts = [f for f in facts if f.volatile]
        stable_facts = [f for f in facts if not f.volatile and f.access_count > 1]
        
        volatility_stability_ratio = len(stable_facts) / max(1, len(volatile_facts))
        
        # Volatility reduction rate (facts that became less volatile over time)
        volatility_reductions = 0
        for fact in facts:
            if len(fact.change_history) >= 2:
                recent_changes = [c for c in fact.change_history 
                                if c.get('timestamp', 0) > recent_cutoff]
                if any('volatility_reduced' in str(c) for c in recent_changes):
                    volatility_reductions += 1
        
        volatility_reduction_rate = volatility_reductions / max(1, len(volatile_facts))
        
        # Calculate consolidation metrics
        consolidated_facts = [f for f in facts 
                            if f.confidence > self.parameters['consolidation_threshold'] and 
                            f.access_count > 2]
        
        consolidation_attempts = len([f for f in facts if f.access_count > 1])
        consolidation_success_rate = len(consolidated_facts) / max(1, consolidation_attempts)
        
        # Average confidence of reinforced beliefs
        reinforced_beliefs = [f for f in facts if f.access_count > 3]
        belief_reinforcement_effectiveness = (
            sum(f.confidence for f in reinforced_beliefs) / max(1, len(reinforced_beliefs))
        )
        
        # User interaction metrics
        if recent_clarifications:
            successful_clarifications = [c for c in recent_clarifications 
                                       if c.resolution_action in ['confirm', 'choose_one', 'update']]
            clarification_success_rate = len(successful_clarifications) / max(1, len(recent_clarifications))
        else:
            clarification_success_rate = 0.5  # Neutral default
        
        # User satisfaction proxy (based on engagement patterns)
        recent_facts = [f for f in facts if f.timestamp and f.timestamp > recent_cutoff]
        avg_access_count = sum(f.access_count for f in recent_facts) / max(1, len(recent_facts))
        user_satisfaction_proxy = min(1.0, avg_access_count / 5.0)  # Normalize to 0-1
        
        # System efficiency metrics
        active_facts = [f for f in facts if f.active]
        memory_utilization = len(active_facts) / max(1, total_facts)
        
        # Query accuracy proxy (based on fact access patterns)
        recently_accessed = [f for f in facts if f.last_accessed and 
                           f.last_accessed > current_time - 3600]  # Last hour
        query_response_accuracy = len(recently_accessed) / max(1, len(facts)) * 10  # Scale up
        query_response_accuracy = min(1.0, query_response_accuracy)
        
        metrics = PerformanceMetrics(
            timestamp=current_time,
            contradiction_detection_rate=contradiction_detection_rate,
            contradiction_resolution_rate=contradiction_resolution_rate,
            false_positive_rate=false_positive_rate,
            volatility_stability_ratio=volatility_stability_ratio,
            volatility_reduction_rate=volatility_reduction_rate,
            consolidation_success_rate=consolidation_success_rate,
            belief_reinforcement_effectiveness=belief_reinforcement_effectiveness,
            clarification_success_rate=clarification_success_rate,
            user_satisfaction_proxy=user_satisfaction_proxy,
            memory_utilization=memory_utilization,
            query_response_accuracy=query_response_accuracy,
            parameters=self.parameters.copy()
        )
        
        self.metrics_history.append(metrics)
        
        print(f"[AutonomousTuning] Collected metrics: "
              f"contradiction_rate={contradiction_detection_rate:.3f}, "
              f"resolution_rate={contradiction_resolution_rate:.3f}, "
              f"stability_ratio={volatility_stability_ratio:.3f}")
        
        return metrics
    
    def should_tune_now(self) -> bool:
        """Check if it's time to perform parameter tuning."""
        if len(self.metrics_history) < self.min_data_points:
            return False
        
        if time.time() - self.last_tuning_time < self.tuning_interval:
            return False
        
        return True
    
    def perform_autonomous_tuning(self) -> List[TuningAction]:
        """
        Analyze performance trends and automatically adjust parameters.
        """
        if not self.should_tune_now():
            return []
        
        print("[AutonomousTuning] Starting autonomous tuning cycle")
        
        tuning_actions = []
        current_metrics = self.metrics_history[-1]
        
        # Analyze trends over stability window
        recent_metrics = list(self.metrics_history)[-self.stability_window:]
        
        # Tune each parameter based on performance indicators
        tuning_actions.extend(self._tune_contradiction_threshold(recent_metrics))
        tuning_actions.extend(self._tune_volatility_parameters(recent_metrics))
        tuning_actions.extend(self._tune_consolidation_parameters(recent_metrics))
        tuning_actions.extend(self._tune_semantic_parameters(recent_metrics))
        tuning_actions.extend(self._tune_clarification_parameters(recent_metrics))
        
        # Apply the tuning actions
        for action in tuning_actions:
            self._apply_tuning_action(action)
            self.tuning_history.append(action)
        
        self.last_tuning_time = time.time()
        
        print(f"[AutonomousTuning] Applied {len(tuning_actions)} parameter adjustments")
        
        return tuning_actions
    
    def _tune_contradiction_threshold(self, recent_metrics: List[PerformanceMetrics]) -> List[TuningAction]:
        """Tune contradiction detection threshold based on accuracy metrics."""
        actions = []
        
        # Calculate average metrics
        avg_false_positive = sum(m.false_positive_rate for m in recent_metrics) / len(recent_metrics)
        avg_resolution_rate = sum(m.contradiction_resolution_rate for m in recent_metrics) / len(recent_metrics)
        
        current_threshold = self.parameters['contradiction_threshold']
        
        # If too many false positives, increase threshold (be more selective)
        if avg_false_positive > 0.3:
            new_threshold = min(
                self.parameter_bounds['contradiction_threshold'][1],
                current_threshold + self.learning_rate * 0.1
            )
            
            if new_threshold != current_threshold:
                actions.append(TuningAction(
                    timestamp=time.time(),
                    parameter_name='contradiction_threshold',
                    old_value=current_threshold,
                    new_value=new_threshold,
                    reason=f"High false positive rate: {avg_false_positive:.3f}",
                    expected_improvement="Reduce false contradiction detections",
                    confidence=0.8
                ))
        
        # If resolution rate is low and false positives aren't too high, lower threshold
        elif avg_resolution_rate < 0.6 and avg_false_positive < 0.2:
            new_threshold = max(
                self.parameter_bounds['contradiction_threshold'][0],
                current_threshold - self.learning_rate * 0.05
            )
            
            if new_threshold != current_threshold:
                actions.append(TuningAction(
                    timestamp=time.time(),
                    parameter_name='contradiction_threshold',
                    old_value=current_threshold,
                    new_value=new_threshold,
                    reason=f"Low resolution rate: {avg_resolution_rate:.3f}",
                    expected_improvement="Detect more resolvable contradictions",
                    confidence=0.6
                ))
        
        return actions
    
    def _tune_volatility_parameters(self, recent_metrics: List[PerformanceMetrics]) -> List[TuningAction]:
        """Tune volatility-related parameters."""
        actions = []
        
        avg_stability_ratio = sum(m.volatility_stability_ratio for m in recent_metrics) / len(recent_metrics)
        avg_volatility_reduction = sum(m.volatility_reduction_rate for m in recent_metrics) / len(recent_metrics)
        
        # If stability ratio is too low (too many volatile facts), increase decay rate
        if avg_stability_ratio < 1.0:  # More volatile than stable facts
            current_decay = self.parameters['volatility_decay_rate']
            new_decay = min(
                self.parameter_bounds['volatility_decay_rate'][1],
                current_decay + self.learning_rate * 0.01
            )
            
            if new_decay != current_decay:
                actions.append(TuningAction(
                    timestamp=time.time(),
                    parameter_name='volatility_decay_rate',
                    old_value=current_decay,
                    new_value=new_decay,
                    reason=f"Low stability ratio: {avg_stability_ratio:.3f}",
                    expected_improvement="Stabilize volatile beliefs faster",
                    confidence=0.7
                ))
        
        return actions
    
    def _tune_consolidation_parameters(self, recent_metrics: List[PerformanceMetrics]) -> List[TuningAction]:
        """Tune belief consolidation parameters."""
        actions = []
        
        avg_consolidation_rate = sum(m.consolidation_success_rate for m in recent_metrics) / len(recent_metrics)
        avg_reinforcement = sum(m.belief_reinforcement_effectiveness for m in recent_metrics) / len(recent_metrics)
        
        # If consolidation rate is low, adjust reinforcement rate
        if avg_consolidation_rate < 0.5:
            current_reinforcement = self.parameters['belief_reinforcement_rate']
            new_reinforcement = min(
                self.parameter_bounds['belief_reinforcement_rate'][1],
                current_reinforcement + self.learning_rate * 0.1
            )
            
            if new_reinforcement != current_reinforcement:
                actions.append(TuningAction(
                    timestamp=time.time(),
                    parameter_name='belief_reinforcement_rate',
                    old_value=current_reinforcement,
                    new_value=new_reinforcement,
                    reason=f"Low consolidation rate: {avg_consolidation_rate:.3f}",
                    expected_improvement="Improve belief consolidation",
                    confidence=0.6
                ))
        
        # Adjust consolidation threshold based on effectiveness
        if avg_reinforcement > 0.9:  # Very high confidence beliefs
            current_threshold = self.parameters['consolidation_threshold']
            new_threshold = min(
                self.parameter_bounds['consolidation_threshold'][1],
                current_threshold + self.learning_rate * 0.02
            )
            
            if new_threshold != current_threshold:
                actions.append(TuningAction(
                    timestamp=time.time(),
                    parameter_name='consolidation_threshold',
                    old_value=current_threshold,
                    new_value=new_threshold,
                    reason=f"High reinforcement effectiveness: {avg_reinforcement:.3f}",
                    expected_improvement="Raise bar for consolidation",
                    confidence=0.5
                ))
        
        return actions
    
    def _tune_semantic_parameters(self, recent_metrics: List[PerformanceMetrics]) -> List[TuningAction]:
        """Tune semantic similarity thresholds."""
        actions = []
        
        avg_query_accuracy = sum(m.query_response_accuracy for m in recent_metrics) / len(recent_metrics)
        
        # If query accuracy is low, adjust semantic similarity threshold
        if avg_query_accuracy < 0.6:
            current_threshold = self.parameters['semantic_similarity_threshold']
            # Lower threshold to be more inclusive in search
            new_threshold = max(
                self.parameter_bounds['semantic_similarity_threshold'][0],
                current_threshold - self.learning_rate * 0.05
            )
            
            if new_threshold != current_threshold:
                actions.append(TuningAction(
                    timestamp=time.time(),
                    parameter_name='semantic_similarity_threshold',
                    old_value=current_threshold,
                    new_value=new_threshold,
                    reason=f"Low query accuracy: {avg_query_accuracy:.3f}",
                    expected_improvement="Improve semantic search recall",
                    confidence=0.6
                ))
        
        return actions
    
    def _tune_clarification_parameters(self, recent_metrics: List[PerformanceMetrics]) -> List[TuningAction]:
        """Tune clarification system parameters."""
        actions = []
        
        avg_clarification_success = sum(m.clarification_success_rate for m in recent_metrics) / len(recent_metrics)
        avg_user_satisfaction = sum(m.user_satisfaction_proxy for m in recent_metrics) / len(recent_metrics)
        
        # If clarification success is low but user satisfaction is high, 
        # we might be asking too many questions
        if avg_clarification_success < 0.5 and avg_user_satisfaction > 0.7:
            current_threshold = self.parameters['clarification_volatility_threshold']
            new_threshold = min(
                self.parameter_bounds['clarification_volatility_threshold'][1],
                current_threshold + self.learning_rate * 0.05
            )
            
            if new_threshold != current_threshold:
                actions.append(TuningAction(
                    timestamp=time.time(),
                    parameter_name='clarification_volatility_threshold',
                    old_value=current_threshold,
                    new_value=new_threshold,
                    reason=f"Low clarification success: {avg_clarification_success:.3f}, high satisfaction: {avg_user_satisfaction:.3f}",
                    expected_improvement="Reduce unnecessary clarification requests",
                    confidence=0.7
                ))
        
        return actions
    
    def _apply_tuning_action(self, action: TuningAction):
        """Apply a tuning action to update system parameters."""
        self.parameters[action.parameter_name] = action.new_value
        
        print(f"[AutonomousTuning] {action.parameter_name}: "
              f"{action.old_value:.3f} → {action.new_value:.3f} "
              f"(reason: {action.reason})")
    
    def get_current_parameters(self) -> Dict[str, float]:
        """Get current parameter values."""
        return self.parameters.copy()
    
    def set_parameter(self, parameter_name: str, value: float, reason: str = "manual_override"):
        """Manually set a parameter value."""
        if parameter_name not in self.parameters:
            raise ValueError(f"Unknown parameter: {parameter_name}")
        
        bounds = self.parameter_bounds.get(parameter_name, (0.0, 1.0))
        if not (bounds[0] <= value <= bounds[1]):
            raise ValueError(f"Parameter {parameter_name} must be between {bounds[0]} and {bounds[1]}")
        
        old_value = self.parameters[parameter_name]
        self.parameters[parameter_name] = value
        
        # Record the manual change
        action = TuningAction(
            timestamp=time.time(),
            parameter_name=parameter_name,
            old_value=old_value,
            new_value=value,
            reason=reason,
            expected_improvement="Manual adjustment",
            confidence=1.0
        )
        self.tuning_history.append(action)
        
        print(f"[AutonomousTuning] Manual override: {parameter_name} = {value}")
    
    def get_tuning_summary(self) -> Dict[str, Any]:
        """Get summary of tuning activity and current state."""
        recent_actions = [a for a in self.tuning_history 
                         if time.time() - a.timestamp < 24 * 3600]  # Last 24 hours
        
        parameter_changes = defaultdict(list)
        for action in recent_actions:
            parameter_changes[action.parameter_name].append({
                'timestamp': action.timestamp,
                'old_value': action.old_value,
                'new_value': action.new_value,
                'reason': action.reason,
                'confidence': action.confidence
            })
        
        return {
            'current_parameters': self.parameters.copy(),
            'recent_changes': dict(parameter_changes),
            'total_tuning_actions': len(self.tuning_history),
            'last_tuning_time': self.last_tuning_time,
            'metrics_collected': len(self.metrics_history),
            'tuning_enabled': len(self.metrics_history) >= self.min_data_points,
            'next_tuning_in': max(0, self.tuning_interval - (time.time() - self.last_tuning_time)),
            'parameter_bounds': self.parameter_bounds
        }
    
    def export_tuning_data(self, output_file: str) -> bool:
        """Export tuning history and metrics for analysis."""
        try:
            export_data = {
                'parameters': self.parameters,
                'parameter_bounds': self.parameter_bounds,
                'tuning_history': [
                    {
                        'timestamp': action.timestamp,
                        'parameter_name': action.parameter_name,
                        'old_value': action.old_value,
                        'new_value': action.new_value,
                        'reason': action.reason,
                        'expected_improvement': action.expected_improvement,
                        'confidence': action.confidence
                    }
                    for action in self.tuning_history
                ],
                'metrics_history': [
                    {
                        'timestamp': metrics.timestamp,
                        'contradiction_detection_rate': metrics.contradiction_detection_rate,
                        'contradiction_resolution_rate': metrics.contradiction_resolution_rate,
                        'false_positive_rate': metrics.false_positive_rate,
                        'volatility_stability_ratio': metrics.volatility_stability_ratio,
                        'consolidation_success_rate': metrics.consolidation_success_rate,
                        'clarification_success_rate': metrics.clarification_success_rate,
                        'user_satisfaction_proxy': metrics.user_satisfaction_proxy,
                        'query_response_accuracy': metrics.query_response_accuracy,
                        'parameters': metrics.parameters
                    }
                    for metrics in self.metrics_history
                ],
                'export_timestamp': time.time(),
                'tuning_configuration': {
                    'tuning_interval': self.tuning_interval,
                    'min_data_points': self.min_data_points,
                    'learning_rate': self.learning_rate,
                    'stability_window': self.stability_window
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"[AutonomousTuning] Exported tuning data to {output_file}")
            return True
            
        except Exception as e:
            print(f"[AutonomousTuning] Failed to export tuning data: {e}")
            return False 