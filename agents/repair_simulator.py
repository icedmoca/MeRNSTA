#!/usr/bin/env python3
"""
Repair Simulator - Symbolic Repair Path Simulation

This module implements a repair simulator that can predict the outcomes of
different repair strategies using historical reflex score patterns and
symbolic reasoning rules.
"""

import time
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from collections import defaultdict

from storage.reflex_log import ReflexLogger, ReflexScore, get_reflex_logger
from storage.self_model import CognitiveSelfModel


@dataclass
class SimulatedRepair:
    """Represents a simulated repair strategy with predicted outcomes."""
    strategy: str
    predicted_score: float
    predicted_coherence_delta: float
    predicted_volatility_delta: float
    predicted_consistency_delta: float
    confidence: float  # Confidence in the prediction (0.0 to 1.0)
    reasoning_chain: List[str]  # Symbolic reasoning steps
    estimated_duration: float  # Estimated execution time
    risk_factors: List[str]  # Potential risk factors
    success_probability: float  # Probability of success
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulatedRepair':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class RepairSimulationResult:
    """Result of a repair path simulation."""
    goal_id: str
    current_state: Dict[str, Any]
    simulated_repairs: List[SimulatedRepair]
    best_repair: Optional[SimulatedRepair]
    reasoning_summary: str
    simulation_timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['simulated_repairs'] = [r.to_dict() for r in self.simulated_repairs]
        if self.best_repair:
            data['best_repair'] = self.best_repair.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RepairSimulationResult':
        """Create from dictionary."""
        repairs_data = data.pop('simulated_repairs', [])
        best_repair_data = data.pop('best_repair', None)
        
        result = cls(**data)
        result.simulated_repairs = [SimulatedRepair.from_dict(r) for r in repairs_data]
        if best_repair_data:
            result.best_repair = SimulatedRepair.from_dict(best_repair_data)
        return result


class RepairSimulator:
    """
    Simulates repair strategies and predicts their outcomes.
    
    Features:
    - Simulates multiple possible repair strategies
    - Estimates outcomes using historical reflex score patterns
    - Uses symbolic reasoning for prediction confidence
    - Provides detailed reasoning traces
    - Identifies risk factors and success probabilities
    """
    
    def __init__(self, reflex_logger: Optional[ReflexLogger] = None, 
                 self_model: Optional[CognitiveSelfModel] = None):
        self.reflex_logger = reflex_logger or get_reflex_logger()
        self.self_model = self_model or CognitiveSelfModel()
        
        # Simulation parameters
        self.min_confidence_threshold = 0.3
        self.max_simulation_depth = 3
        self.strategy_weights = {
            'historical_performance': 0.4,
            'symbolic_rules': 0.3,
            'context_similarity': 0.2,
            'risk_assessment': 0.1
        }
        
        # Available repair strategies
        self.available_strategies = [
            'belief_clarification',
            'cluster_reassessment', 
            'fact_consolidation',
            'anticipatory_drift'
        ]
        
        print(f"[RepairSimulator] Initialized repair simulator")
    
    def simulate_repair_paths(self, goal_id: str, current_state: Dict[str, Any], 
                            target_strategies: Optional[List[str]] = None) -> RepairSimulationResult:
        """
        Simulate multiple repair strategies for a given goal and current state.
        
        Args:
            goal_id: Identifier for the goal being repaired
            current_state: Current cognitive state
            target_strategies: Specific strategies to simulate (None for all)
            
        Returns:
            RepairSimulationResult with simulated outcomes
        """
        try:
            strategies = target_strategies or self.available_strategies
            simulated_repairs = []
            
            for strategy in strategies:
                simulated_repair = self._simulate_single_strategy(
                    strategy, goal_id, current_state
                )
                if simulated_repair:
                    simulated_repairs.append(simulated_repair)
            
            # Sort by predicted score
            simulated_repairs.sort(key=lambda x: x.predicted_score, reverse=True)
            
            # Determine best repair
            best_repair = simulated_repairs[0] if simulated_repairs else None
            
            # Generate reasoning summary
            reasoning_summary = self._generate_reasoning_summary(simulated_repairs, best_repair)
            
            return RepairSimulationResult(
                goal_id=goal_id,
                current_state=current_state,
                simulated_repairs=simulated_repairs,
                best_repair=best_repair,
                reasoning_summary=reasoning_summary,
                simulation_timestamp=time.time()
            )
            
        except Exception as e:
            logging.error(f"[RepairSimulator] Error simulating repair paths: {e}")
            return RepairSimulationResult(
                goal_id=goal_id,
                current_state=current_state,
                simulated_repairs=[],
                best_repair=None,
                reasoning_summary=f"Simulation failed: {e}",
                simulation_timestamp=time.time()
            )
    
    def _simulate_single_strategy(self, strategy: str, goal_id: str, 
                                current_state: Dict[str, Any]) -> Optional[SimulatedRepair]:
        """
        Simulate a single repair strategy.
        
        Args:
            strategy: Strategy to simulate
            goal_id: Goal identifier
            current_state: Current cognitive state
            
        Returns:
            SimulatedRepair with predicted outcomes
        """
        try:
            # Get historical performance data
            historical_scores = self.reflex_logger.get_scores_by_strategy(strategy, limit=20)
            
            if not historical_scores:
                # No historical data, use default predictions
                return self._create_default_simulation(strategy, current_state)
            
            # Calculate base predictions from historical data
            base_predictions = self._calculate_base_predictions(historical_scores)
            
            # Apply symbolic reasoning adjustments
            symbolic_adjustments = self._apply_symbolic_reasoning(strategy, current_state)
            
            # Apply context similarity adjustments
            context_adjustments = self._apply_context_similarity(strategy, current_state, historical_scores)
            
            # Calculate risk factors
            risk_factors = self._identify_risk_factors(strategy, current_state, historical_scores)
            
            # Combine predictions
            final_predictions = self._combine_predictions(
                base_predictions, symbolic_adjustments, context_adjustments
            )
            
            # Generate reasoning chain
            reasoning_chain = self._generate_reasoning_chain(
                strategy, base_predictions, symbolic_adjustments, 
                context_adjustments, risk_factors
            )
            
            # Calculate confidence
            confidence = self._calculate_prediction_confidence(
                historical_scores, symbolic_adjustments, context_adjustments
            )
            
            # Calculate success probability
            success_probability = self._calculate_success_probability(
                historical_scores, risk_factors, confidence
            )
            
            return SimulatedRepair(
                strategy=strategy,
                predicted_score=final_predictions['score'],
                predicted_coherence_delta=final_predictions['coherence_delta'],
                predicted_volatility_delta=final_predictions['volatility_delta'],
                predicted_consistency_delta=final_predictions['consistency_delta'],
                confidence=confidence,
                reasoning_chain=reasoning_chain,
                estimated_duration=self._estimate_duration(strategy, historical_scores),
                risk_factors=risk_factors,
                success_probability=success_probability
            )
            
        except Exception as e:
            logging.error(f"[RepairSimulator] Error simulating strategy {strategy}: {e}")
            return None
    
    def _calculate_base_predictions(self, historical_scores: List[ReflexScore]) -> Dict[str, float]:
        """Calculate base predictions from historical reflex scores."""
        if not historical_scores:
            return {
                'score': 0.5,
                'coherence_delta': 0.0,
                'volatility_delta': 0.0,
                'consistency_delta': 0.0
            }
        
        # Calculate weighted averages (more recent scores have higher weight)
        weights = np.linspace(0.5, 1.0, len(historical_scores))
        weights = weights / np.sum(weights)
        
        return {
            'score': np.average([s.score for s in historical_scores], weights=weights),
            'coherence_delta': np.average([s.coherence_delta for s in historical_scores], weights=weights),
            'volatility_delta': np.average([s.volatility_delta for s in historical_scores], weights=weights),
            'consistency_delta': np.average([s.belief_consistency_delta for s in historical_scores], weights=weights)
        }
    
    def _apply_symbolic_reasoning(self, strategy: str, current_state: Dict[str, Any]) -> Dict[str, float]:
        """Apply symbolic reasoning adjustments to predictions."""
        adjustments = {
            'score': 0.0,
            'coherence_delta': 0.0,
            'volatility_delta': 0.0,
            'consistency_delta': 0.0
        }
        
        try:
            # Query symbolic rules from self-model
            rules = self.self_model.query_rules(f"Strategy({strategy})")
            
            for rule in rules:
                if rule.confidence > self.min_confidence_threshold:
                    # Apply rule-based adjustments
                    if "coherence" in rule.consequent.lower():
                        adjustments['coherence_delta'] += 0.1 * rule.confidence
                    if "volatility" in rule.consequent.lower():
                        adjustments['volatility_delta'] -= 0.1 * rule.confidence
                    if "consistency" in rule.consequent.lower():
                        adjustments['consistency_delta'] += 0.1 * rule.confidence
                    if "score" in rule.consequent.lower():
                        adjustments['score'] += 0.1 * rule.confidence
            
            # Context-specific adjustments
            drift_type = current_state.get('drift_type', 'unknown')
            if drift_type == 'contradiction' and strategy == 'belief_clarification':
                adjustments['consistency_delta'] += 0.2
                adjustments['score'] += 0.15
            elif drift_type == 'volatility' and strategy == 'cluster_reassessment':
                adjustments['volatility_delta'] -= 0.2
                adjustments['score'] += 0.15
            elif drift_type == 'coherence' and strategy == 'fact_consolidation':
                adjustments['coherence_delta'] += 0.2
                adjustments['score'] += 0.15
            
        except Exception as e:
            logging.warning(f"[RepairSimulator] Error applying symbolic reasoning: {e}")
        
        return adjustments
    
    def _apply_context_similarity(self, strategy: str, current_state: Dict[str, Any], 
                                historical_scores: List[ReflexScore]) -> Dict[str, float]:
        """Apply context similarity adjustments."""
        adjustments = {
            'score': 0.0,
            'coherence_delta': 0.0,
            'volatility_delta': 0.0,
            'consistency_delta': 0.0
        }
        
        try:
            # Find similar historical contexts
            current_cluster_id = current_state.get('cluster_id')
            current_drift_score = current_state.get('drift_score', 0.0)
            
            similar_scores = []
            for score in historical_scores:
                similarity = 0.0
                
                # Cluster similarity
                if score.cluster_id == current_cluster_id:
                    similarity += 0.5
                
                # Drift score similarity
                drift_similarity = 1.0 - abs(score.score - current_drift_score)
                similarity += 0.3 * max(0, drift_similarity)
                
                # Token similarity
                if score.token_id and current_state.get('token_id'):
                    if score.token_id == current_state['token_id']:
                        similarity += 0.2
                
                if similarity > 0.3:  # Threshold for similarity
                    similar_scores.append((score, similarity))
            
            # Apply weighted adjustments based on similar contexts
            if similar_scores:
                total_weight = sum(weight for _, weight in similar_scores)
                for score, weight in similar_scores:
                    weight_ratio = weight / total_weight
                    adjustments['score'] += score.score * weight_ratio * 0.1
                    adjustments['coherence_delta'] += score.coherence_delta * weight_ratio * 0.1
                    adjustments['volatility_delta'] += score.volatility_delta * weight_ratio * 0.1
                    adjustments['consistency_delta'] += score.belief_consistency_delta * weight_ratio * 0.1
        
        except Exception as e:
            logging.warning(f"[RepairSimulator] Error applying context similarity: {e}")
        
        return adjustments
    
    def _identify_risk_factors(self, strategy: str, current_state: Dict[str, Any], 
                             historical_scores: List[ReflexScore]) -> List[str]:
        """Identify potential risk factors for the strategy."""
        risk_factors = []
        
        try:
            # Low historical success rate
            if historical_scores:
                success_rate = sum(1 for s in historical_scores if s.success) / len(historical_scores)
                if success_rate < 0.5:
                    risk_factors.append(f"Low historical success rate ({success_rate:.2f})")
            
            # High volatility in current state
            current_volatility = current_state.get('volatility_score', 0.0)
            if current_volatility > 0.7:
                risk_factors.append(f"High current volatility ({current_volatility:.2f})")
            
            # Contradiction density
            contradiction_count = current_state.get('contradiction_count', 0)
            if contradiction_count > 5:
                risk_factors.append(f"High contradiction density ({contradiction_count})")
            
            # Strategy-specific risks
            if strategy == 'belief_clarification':
                if current_state.get('belief_confidence', 1.0) < 0.3:
                    risk_factors.append("Low belief confidence may limit clarification effectiveness")
            
            elif strategy == 'cluster_reassessment':
                cluster_size = current_state.get('cluster_size', 0)
                if cluster_size > 20:
                    risk_factors.append(f"Large cluster size ({cluster_size}) may slow reassessment")
            
            elif strategy == 'fact_consolidation':
                fact_count = current_state.get('affected_fact_count', 0)
                if fact_count > 10:
                    risk_factors.append(f"Many affected facts ({fact_count}) may complicate consolidation")
        
        except Exception as e:
            logging.warning(f"[RepairSimulator] Error identifying risk factors: {e}")
            risk_factors.append(f"Error in risk assessment: {e}")
        
        return risk_factors
    
    def _combine_predictions(self, base_predictions: Dict[str, float], 
                           symbolic_adjustments: Dict[str, float],
                           context_adjustments: Dict[str, float]) -> Dict[str, float]:
        """Combine different prediction components using weights."""
        weights = self.strategy_weights
        
        combined = {}
        for key in base_predictions:
            combined[key] = (
                base_predictions[key] * weights['historical_performance'] +
                symbolic_adjustments[key] * weights['symbolic_rules'] +
                context_adjustments[key] * weights['context_similarity']
            )
        
        # Ensure predictions are within reasonable bounds
        combined['score'] = max(0.0, min(1.0, combined['score']))
        combined['coherence_delta'] = max(-1.0, min(1.0, combined['coherence_delta']))
        combined['volatility_delta'] = max(-1.0, min(1.0, combined['volatility_delta']))
        combined['consistency_delta'] = max(-1.0, min(1.0, combined['consistency_delta']))
        
        return combined
    
    def _generate_reasoning_chain(self, strategy: str, base_predictions: Dict[str, float],
                                symbolic_adjustments: Dict[str, float],
                                context_adjustments: Dict[str, float],
                                risk_factors: List[str]) -> List[str]:
        """Generate a reasoning chain explaining the simulation."""
        chain = []
        
        # Base reasoning
        chain.append(f"Strategy: {strategy}")
        chain.append(f"Base score prediction: {base_predictions['score']:.3f}")
        
        # Symbolic reasoning
        if any(abs(v) > 0.01 for v in symbolic_adjustments.values()):
            chain.append("Symbolic reasoning adjustments applied:")
            for key, value in symbolic_adjustments.items():
                if abs(value) > 0.01:
                    chain.append(f"  {key}: {value:+.3f}")
        
        # Context similarity
        if any(abs(v) > 0.01 for v in context_adjustments.values()):
            chain.append("Context similarity adjustments applied:")
            for key, value in context_adjustments.items():
                if abs(value) > 0.01:
                    chain.append(f"  {key}: {value:+.3f}")
        
        # Risk factors
        if risk_factors:
            chain.append("Identified risk factors:")
            for risk in risk_factors:
                chain.append(f"  - {risk}")
        
        return chain
    
    def _calculate_prediction_confidence(self, historical_scores: List[ReflexScore],
                                       symbolic_adjustments: Dict[str, float],
                                       context_adjustments: Dict[str, float]) -> float:
        """Calculate confidence in the prediction."""
        confidence = 0.5  # Base confidence
        
        # Historical data confidence
        if historical_scores:
            # More data points increase confidence
            data_confidence = min(1.0, len(historical_scores) / 10.0)
            confidence += 0.3 * data_confidence
            
            # Consistency in historical data
            if len(historical_scores) > 1:
                scores = [s.score for s in historical_scores]
                consistency = 1.0 - np.std(scores)
                confidence += 0.2 * max(0, consistency)
        
        # Symbolic reasoning confidence
        symbolic_strength = sum(abs(v) for v in symbolic_adjustments.values())
        confidence += 0.2 * min(1.0, symbolic_strength)
        
        # Context similarity confidence
        context_strength = sum(abs(v) for v in context_adjustments.values())
        confidence += 0.1 * min(1.0, context_strength)
        
        return min(1.0, confidence)
    
    def _calculate_success_probability(self, historical_scores: List[ReflexScore],
                                     risk_factors: List[str], confidence: float) -> float:
        """Calculate probability of success."""
        if not historical_scores:
            return 0.5 * confidence
        
        # Base success rate from history
        success_rate = sum(1 for s in historical_scores if s.success) / len(historical_scores)
        
        # Adjust for confidence
        adjusted_rate = success_rate * confidence
        
        # Adjust for risk factors
        risk_penalty = len(risk_factors) * 0.05
        adjusted_rate = max(0.0, adjusted_rate - risk_penalty)
        
        return min(1.0, adjusted_rate)
    
    def _estimate_duration(self, strategy: str, historical_scores: List[ReflexScore]) -> float:
        """Estimate execution duration based on historical data."""
        if not historical_scores:
            # Default durations
            default_durations = {
                'belief_clarification': 30.0,
                'cluster_reassessment': 45.0,
                'fact_consolidation': 60.0,
                'anticipatory_drift': 20.0
            }
            return default_durations.get(strategy, 30.0)
        
        # Calculate average duration from historical data
        durations = []
        for score in historical_scores:
            # Extract duration from scoring notes or use default
            if hasattr(score, 'duration') and score.duration > 0:
                durations.append(score.duration)
        
        if durations:
            return np.mean(durations)
        else:
            return 30.0  # Default duration
    
    def _create_default_simulation(self, strategy: str, current_state: Dict[str, Any]) -> SimulatedRepair:
        """Create a default simulation when no historical data is available."""
        return SimulatedRepair(
            strategy=strategy,
            predicted_score=0.5,
            predicted_coherence_delta=0.0,
            predicted_volatility_delta=0.0,
            predicted_consistency_delta=0.0,
            confidence=0.3,
            reasoning_chain=[
                f"Strategy: {strategy}",
                "No historical data available",
                "Using default predictions"
            ],
            estimated_duration=30.0,
            risk_factors=["No historical performance data"],
            success_probability=0.5
        )
    
    def _generate_reasoning_summary(self, simulated_repairs: List[SimulatedRepair], 
                                  best_repair: Optional[SimulatedRepair]) -> str:
        """Generate a summary of the simulation reasoning."""
        if not simulated_repairs:
            return "No repair strategies could be simulated."
        
        summary = f"Simulated {len(simulated_repairs)} repair strategies:\n"
        
        for i, repair in enumerate(simulated_repairs[:3], 1):  # Top 3
            summary += f"\n{i}. {repair.strategy} (Score: {repair.predicted_score:.3f}, Confidence: {repair.confidence:.3f})"
            if repair.risk_factors:
                summary += f"\n   Risks: {', '.join(repair.risk_factors[:2])}"
        
        if best_repair:
            summary += f"\n\nBest strategy: {best_repair.strategy}"
            summary += f"\nReasoning: {best_repair.reasoning_chain[0] if best_repair.reasoning_chain else 'No reasoning available'}"
        
        return summary


# Global repair simulator instance
_repair_simulator_instance = None


def get_repair_simulator() -> RepairSimulator:
    """Get or create the global repair simulator instance."""
    global _repair_simulator_instance
    
    if _repair_simulator_instance is None:
        _repair_simulator_instance = RepairSimulator()
    
    return _repair_simulator_instance


def simulate_repair_paths(goal_id: str, current_state: Dict[str, Any], 
                         target_strategies: Optional[List[str]] = None) -> RepairSimulationResult:
    """Convenience function to simulate repair paths."""
    return get_repair_simulator().simulate_repair_paths(goal_id, current_state, target_strategies) 