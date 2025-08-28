#!/usr/bin/env python3
"""
StrategyEvaluator - Advanced Strategy Comparison and Scoring for MeRNSTA

Provides sophisticated evaluation of planning strategies using multiple criteria,
historical data analysis, and adaptive scoring based on context and outcomes.
"""

import logging
import json
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .base import BaseAgent


class EvaluationCriteria(Enum):
    """Criteria for strategy evaluation"""
    FEASIBILITY = "feasibility"
    COMPLEXITY = "complexity"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    TIME_TO_COMPLETION = "time_to_completion"
    SUCCESS_PROBABILITY = "success_probability"
    RISK_MITIGATION = "risk_mitigation"
    ADAPTABILITY = "adaptability"
    PARALLEL_POTENTIAL = "parallel_potential"
    RECOVERY_OPTIONS = "recovery_options"
    LEARNING_VALUE = "learning_value"


@dataclass
class EvaluationResult:
    """Result of strategy evaluation"""
    strategy_id: str
    overall_score: float
    criteria_scores: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    confidence: float
    evaluation_timestamp: datetime
    context_factors: Dict[str, Any]


@dataclass
class ComparisonResult:
    """Result comparing multiple strategies"""
    best_strategy_id: str
    strategy_rankings: List[Tuple[str, float]]
    evaluation_results: List[EvaluationResult]
    comparison_summary: Dict[str, Any]
    decision_rationale: str


class StrategyEvaluator(BaseAgent):
    """
    Advanced strategy evaluation and comparison system.
    
    Capabilities:
    - Multi-criteria strategy scoring
    - Historical performance analysis
    - Context-aware evaluation
    - Adaptive scoring weights
    - Strategy comparison and ranking
    - Recommendation generation
    """
    
    def __init__(self):
        super().__init__("strategy_evaluator")
        
        # Load evaluation configuration
        self.config = self._load_evaluation_config()
        self.enabled = self.config.get('enabled', True)
        
        # Scoring weights (can be adapted based on context)
        self.base_weights = self.config.get('base_weights', {
            EvaluationCriteria.FEASIBILITY.value: 1.0,
            EvaluationCriteria.COMPLEXITY.value: 0.8,
            EvaluationCriteria.RESOURCE_EFFICIENCY.value: 0.9,
            EvaluationCriteria.TIME_TO_COMPLETION.value: 0.7,
            EvaluationCriteria.SUCCESS_PROBABILITY.value: 1.2,
            EvaluationCriteria.RISK_MITIGATION.value: 0.8,
            EvaluationCriteria.ADAPTABILITY.value: 0.6,
            EvaluationCriteria.PARALLEL_POTENTIAL.value: 0.5,
            EvaluationCriteria.RECOVERY_OPTIONS.value: 0.7,
            EvaluationCriteria.LEARNING_VALUE.value: 0.4
        })
        
        # Evaluation parameters
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.historical_weight = self.config.get('historical_weight', 0.3)
        self.context_adaptation_rate = self.config.get('context_adaptation_rate', 0.1)
        
        # Historical data
        self.strategy_history: Dict[str, List[Dict[str, Any]]] = {}
        self.evaluation_history: List[EvaluationResult] = []
        self.context_patterns: Dict[str, Dict[str, float]] = {}
        
        logging.info(f"[{self.name}] Initialized strategy evaluator")
    
    def get_agent_instructions(self) -> str:
        """Return specialized instructions for the strategy evaluator agent."""
        return (
            "You are a strategy evaluation specialist focused on analyzing and scoring strategic approaches. "
            "Your role is to assess strategies across multiple criteria including feasibility, complexity, "
            "resource efficiency, success probability, and risk factors. You provide comparative analysis, "
            "ranking recommendations, and adaptive scoring based on context. Focus on objective evaluation, "
            "evidence-based assessment, and helping decision-makers choose optimal strategies for their goals."
        )
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """Generate strategy evaluation responses and assess strategic options."""
        context = context or {}
        
        # Build memory context for evaluation patterns
        memory_context = self.get_memory_context(message)
        
        # Use LLM if available for complex evaluation questions
        if self.llm_fallback:
            prompt = self.build_agent_prompt(message, memory_context)
            try:
                return self.llm_fallback.process(prompt)
            except Exception as e:
                logging.error(f"[{self.name}] LLM processing failed: {e}")
        
        # Handle strategy evaluation requests
        if "evaluate" in message.lower() or "assess" in message.lower():
            if "strategy" in context:
                try:
                    result = self.evaluate_strategy(context["strategy"], context.get("evaluation_context"))
                    return f"Strategy evaluation complete. Score: {result.overall_score:.2f}, Confidence: {result.confidence:.2f}"
                except Exception as e:
                    return f"Strategy evaluation failed: {str(e)}"
            else:
                return "Please provide a strategy to evaluate."
        
        if "compare" in message.lower() and "strategies" in context:
            try:
                strategies = context["strategies"]
                results = self.compare_strategies(strategies, context.get("evaluation_context"))
                best = max(results, key=lambda x: x.overall_score)
                return f"Compared {len(strategies)} strategies. Best: {best.strategy_id} (score: {best.overall_score:.2f})"
            except Exception as e:
                return f"Strategy comparison failed: {str(e)}"
        
        return "I can evaluate and compare strategies across multiple criteria. Provide strategies to assess or ask about evaluation methods."

    def evaluate_strategy(self, strategy: Any, context: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """
        Evaluate a single strategy across multiple criteria.
        
        Args:
            strategy: Strategy object to evaluate
            context: Evaluation context (urgency, resources, etc.)
            
        Returns:
            Detailed evaluation result
        """
        if not self.enabled:
            return self._create_default_evaluation(strategy.strategy_id)
        
        context = context or {}
        logging.info(f"[{self.name}] Evaluating strategy {strategy.strategy_id}")
        
        # Calculate scores for each criterion
        criteria_scores = {}
        for criterion in EvaluationCriteria:
            criteria_scores[criterion.value] = self._evaluate_criterion(strategy, criterion, context)
        
        # Calculate overall score with adaptive weights
        weights = self._get_adaptive_weights(context)
        overall_score = self._calculate_weighted_score(criteria_scores, weights)
        
        # Identify strengths and weaknesses
        strengths, weaknesses = self._analyze_strategy_profile(criteria_scores)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(strategy, criteria_scores, context)
        
        # Calculate confidence
        confidence = self._calculate_evaluation_confidence(strategy, criteria_scores, context)
        
        # Create evaluation result
        evaluation = EvaluationResult(
            strategy_id=strategy.strategy_id,
            overall_score=overall_score,
            criteria_scores=criteria_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            confidence=confidence,
            evaluation_timestamp=datetime.now(),
            context_factors=context
        )
        
        # Store for historical analysis
        self.evaluation_history.append(evaluation)
        self._update_historical_data(strategy, evaluation)
        
        return evaluation
    
    def compare_strategies(self, strategies: List[Any], context: Optional[Dict[str, Any]] = None) -> ComparisonResult:
        """
        Compare multiple strategies and rank them.
        
        Args:
            strategies: List of strategies to compare
            context: Evaluation context
            
        Returns:
            Comparison result with rankings and analysis
        """
        if not strategies:
            raise ValueError("No strategies provided for comparison")
        
        logging.info(f"[{self.name}] Comparing {len(strategies)} strategies")
        
        # Evaluate each strategy
        evaluations = []
        for strategy in strategies:
            evaluation = self.evaluate_strategy(strategy, context)
            evaluations.append(evaluation)
        
        # Rank strategies by overall score
        rankings = sorted(
            [(eval.strategy_id, eval.overall_score) for eval in evaluations],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Generate comparison summary
        summary = self._generate_comparison_summary(evaluations, context)
        
        # Generate decision rationale
        rationale = self._generate_decision_rationale(evaluations, rankings, context)
        
        return ComparisonResult(
            best_strategy_id=rankings[0][0],
            strategy_rankings=rankings,
            evaluation_results=evaluations,
            comparison_summary=summary,
            decision_rationale=rationale
        )
    
    def adapt_weights_from_feedback(self, strategy_id: str, outcome: Dict[str, Any]) -> None:
        """
        Adapt evaluation weights based on strategy execution outcomes.
        
        Args:
            strategy_id: ID of the executed strategy
            outcome: Execution outcome and performance data
        """
        logging.info(f"[{self.name}] Adapting weights based on outcome for {strategy_id}")
        
        # Find the evaluation for this strategy
        evaluation = None
        for eval_result in self.evaluation_history:
            if eval_result.strategy_id == strategy_id:
                evaluation = eval_result
                break
        
        if not evaluation:
            logging.warning(f"[{self.name}] No evaluation found for strategy {strategy_id}")
            return
        
        # Analyze outcome vs prediction
        success = outcome.get('success', False)
        actual_duration = outcome.get('duration_minutes', 0)
        actual_resources = outcome.get('resources_used', {})
        
        # Update weights based on prediction accuracy
        self._update_weights_from_outcome(evaluation, outcome)
        
        # Store outcome for future reference
        self._store_outcome_data(strategy_id, evaluation, outcome)
    
    def get_evaluation_insights(self) -> Dict[str, Any]:
        """
        Get insights from evaluation history and patterns.
        
        Returns:
            Analysis of evaluation patterns and trends
        """
        if not self.evaluation_history:
            return {"message": "No evaluation history available"}
        
        insights = {
            "total_evaluations": len(self.evaluation_history),
            "average_scores": self._calculate_average_scores(),
            "criteria_trends": self._analyze_criteria_trends(),
            "context_patterns": self._analyze_context_patterns(),
            "weight_adaptations": self._analyze_weight_adaptations(),
            "prediction_accuracy": self._calculate_prediction_accuracy(),
            "recommendations": self._generate_system_recommendations()
        }
        
        return insights
    
    def _evaluate_criterion(self, strategy: Any, criterion: EvaluationCriteria, context: Dict[str, Any]) -> float:
        """Evaluate a strategy against a specific criterion."""
        if criterion == EvaluationCriteria.FEASIBILITY:
            return self._evaluate_feasibility(strategy, context)
        elif criterion == EvaluationCriteria.COMPLEXITY:
            return self._evaluate_complexity(strategy, context)
        elif criterion == EvaluationCriteria.RESOURCE_EFFICIENCY:
            return self._evaluate_resource_efficiency(strategy, context)
        elif criterion == EvaluationCriteria.TIME_TO_COMPLETION:
            return self._evaluate_time_efficiency(strategy, context)
        elif criterion == EvaluationCriteria.SUCCESS_PROBABILITY:
            return self._evaluate_success_probability(strategy, context)
        elif criterion == EvaluationCriteria.RISK_MITIGATION:
            return self._evaluate_risk_mitigation(strategy, context)
        elif criterion == EvaluationCriteria.ADAPTABILITY:
            return self._evaluate_adaptability(strategy, context)
        elif criterion == EvaluationCriteria.PARALLEL_POTENTIAL:
            return self._evaluate_parallel_potential(strategy, context)
        elif criterion == EvaluationCriteria.RECOVERY_OPTIONS:
            return self._evaluate_recovery_options(strategy, context)
        elif criterion == EvaluationCriteria.LEARNING_VALUE:
            return self._evaluate_learning_value(strategy, context)
        else:
            return 0.5  # Default neutral score
    
    def _evaluate_feasibility(self, strategy: Any, context: Dict[str, Any]) -> float:
        """Evaluate how feasible the strategy is to execute."""
        # Consider factors like available resources, prerequisites, etc.
        base_feasibility = 0.8  # Start with optimistic assumption
        
        # Adjust based on resource requirements
        resource_availability = context.get('resource_availability', 1.0)
        base_feasibility *= resource_availability
        
        # Adjust based on complexity
        step_count = len(strategy.nodes) if hasattr(strategy, 'nodes') else len(strategy.plan.steps)
        complexity_penalty = min(0.3, (step_count - 3) * 0.05)
        base_feasibility -= complexity_penalty
        
        # Adjust based on confidence levels
        if hasattr(strategy, 'nodes'):
            avg_confidence = sum(node.step.confidence for node in strategy.nodes) / len(strategy.nodes)
        else:
            avg_confidence = sum(step.confidence for step in strategy.plan.steps) / len(strategy.plan.steps)
        
        base_feasibility = (base_feasibility + avg_confidence) / 2
        
        return max(0.0, min(1.0, base_feasibility))
    
    def _evaluate_complexity(self, strategy: Any, context: Dict[str, Any]) -> float:
        """Evaluate strategy complexity (lower complexity = higher score)."""
        if hasattr(strategy, 'nodes'):
            step_count = len(strategy.nodes)
            dependency_count = sum(len(node.dependencies) for node in strategy.nodes)
        else:
            step_count = len(strategy.plan.steps)
            dependency_count = sum(len(step.prerequisites) for step in strategy.plan.steps)
        
        # Normalize complexity score (simpler = higher score)
        complexity_score = 1.0 - min(1.0, (step_count + dependency_count) / 20)
        
        return max(0.0, complexity_score)
    
    def _evaluate_resource_efficiency(self, strategy: Any, context: Dict[str, Any]) -> float:
        """Evaluate how efficiently the strategy uses resources."""
        # Consider time, computational resources, external dependencies
        total_duration = 0
        if hasattr(strategy, 'nodes'):
            total_duration = sum(node.estimated_duration or 30 for node in strategy.nodes)
        else:
            total_duration = 60 * len(strategy.plan.steps)  # Estimate 60 min per step
        
        # Efficiency decreases with duration (up to a point)
        max_reasonable_duration = 480  # 8 hours
        efficiency_score = max(0.0, 1.0 - (total_duration / max_reasonable_duration))
        
        return efficiency_score
    
    def _evaluate_time_efficiency(self, strategy: Any, context: Dict[str, Any]) -> float:
        """Evaluate how quickly the strategy can be completed."""
        urgency = context.get('urgency', 0.5)  # 0 = not urgent, 1 = very urgent
        
        if hasattr(strategy, 'estimated_completion'):
            time_to_completion = (strategy.estimated_completion - datetime.now()).total_seconds() / 3600  # hours
        else:
            # Estimate based on steps
            step_count = len(strategy.plan.steps)
            time_to_completion = step_count * 1  # 1 hour per step estimate
        
        # Score based on urgency and time
        if urgency > 0.7:  # High urgency
            time_score = max(0.0, 1.0 - (time_to_completion / 4))  # Penalize if > 4 hours
        else:  # Lower urgency
            time_score = max(0.0, 1.0 - (time_to_completion / 24))  # Penalize if > 1 day
        
        return time_score
    
    def _evaluate_success_probability(self, strategy: Any, context: Dict[str, Any]) -> float:
        """Evaluate the probability of strategy success."""
        if hasattr(strategy, 'success_probability'):
            base_probability = strategy.success_probability
        else:
            # Calculate based on step confidences
            if hasattr(strategy, 'nodes'):
                confidences = [node.step.confidence for node in strategy.nodes]
            else:
                confidences = [step.confidence for step in strategy.plan.steps]
            
            # Success probability is the product of step confidences
            base_probability = 1.0
            for confidence in confidences:
                base_probability *= confidence
        
        # Adjust based on historical performance
        historical_adjustment = self._get_historical_success_rate(strategy)
        
        return (base_probability + historical_adjustment) / 2
    
    def _evaluate_risk_mitigation(self, strategy: Any, context: Dict[str, Any]) -> float:
        """Evaluate how well the strategy mitigates risks."""
        risk_factors = strategy.plan.risk_factors or []
        
        # More risk factors = lower score, unless there are mitigation measures
        base_risk_score = max(0.0, 1.0 - len(risk_factors) / 10)
        
        # Check for fallback options
        if hasattr(strategy, 'nodes'):
            fallback_count = sum(len(node.fallback_nodes) for node in strategy.nodes)
            fallback_bonus = min(0.3, fallback_count * 0.1)
            base_risk_score += fallback_bonus
        
        return min(1.0, base_risk_score)
    
    def _evaluate_adaptability(self, strategy: Any, context: Dict[str, Any]) -> float:
        """Evaluate how adaptable the strategy is to changes."""
        # Strategies with more parallel steps and fewer dependencies are more adaptable
        if hasattr(strategy, 'nodes'):
            total_nodes = len(strategy.nodes)
            dependent_nodes = sum(1 for node in strategy.nodes if node.dependencies)
            
            adaptability_score = 1.0 - (dependent_nodes / total_nodes if total_nodes > 0 else 0)
        else:
            # Estimate based on plan type
            if strategy.plan.plan_type == "parallel":
                adaptability_score = 0.9
            elif strategy.plan.plan_type == "tree":
                adaptability_score = 0.7
            else:
                adaptability_score = 0.5
        
        return adaptability_score
    
    def _evaluate_parallel_potential(self, strategy: Any, context: Dict[str, Any]) -> float:
        """Evaluate how much of the strategy can be executed in parallel."""
        if hasattr(strategy, 'nodes'):
            total_nodes = len(strategy.nodes)
            nodes_with_deps = sum(1 for node in strategy.nodes if node.dependencies)
            
            # More independent nodes = higher parallel potential
            parallel_score = 1.0 - (nodes_with_deps / total_nodes if total_nodes > 0 else 0)
        else:
            # Estimate based on plan type
            type_scores = {
                "parallel": 0.9,
                "tree": 0.7,
                "sequential": 0.2,
                "dag": 0.8
            }
            parallel_score = type_scores.get(strategy.plan.plan_type, 0.5)
        
        return parallel_score
    
    def _evaluate_recovery_options(self, strategy: Any, context: Dict[str, Any]) -> float:
        """Evaluate recovery options if the strategy fails."""
        if hasattr(strategy, 'nodes'):
            total_nodes = len(strategy.nodes)
            nodes_with_fallbacks = sum(1 for node in strategy.nodes if node.fallback_nodes)
            
            recovery_score = nodes_with_fallbacks / total_nodes if total_nodes > 0 else 0
        else:
            # Conservative estimate
            recovery_score = 0.3
        
        # Bonus for strategies that allow partial success
        if strategy.strategy_type.value in ["tree", "parallel", "dag"]:
            recovery_score += 0.2
        
        return min(1.0, recovery_score)
    
    def _evaluate_learning_value(self, strategy: Any, context: Dict[str, Any]) -> float:
        """Evaluate how much can be learned from executing this strategy."""
        # Novel strategies have higher learning value
        novelty_score = 1.0 - self._get_strategy_similarity_to_history(strategy)
        
        # Strategies with more diverse steps have higher learning value
        if hasattr(strategy, 'nodes'):
            step_types = set(node.step.subgoal.split()[0].lower() for node in strategy.nodes)
            diversity_score = min(1.0, len(step_types) / 5)
        else:
            diversity_score = 0.5
        
        return (novelty_score + diversity_score) / 2
    
    def _get_adaptive_weights(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Get weights adapted to the current context."""
        weights = self.base_weights.copy()
        
        # Adapt based on context
        urgency = context.get('urgency', 0.5)
        if urgency > 0.7:
            # High urgency: emphasize time and feasibility
            weights[EvaluationCriteria.TIME_TO_COMPLETION.value] *= 1.5
            weights[EvaluationCriteria.FEASIBILITY.value] *= 1.3
            weights[EvaluationCriteria.LEARNING_VALUE.value] *= 0.5
        
        resource_constraints = context.get('resource_constraints', 0.5)
        if resource_constraints > 0.7:
            # High resource constraints: emphasize efficiency
            weights[EvaluationCriteria.RESOURCE_EFFICIENCY.value] *= 1.4
            weights[EvaluationCriteria.COMPLEXITY.value] *= 1.2
        
        return weights
    
    def _calculate_weighted_score(self, criteria_scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted overall score."""
        total_score = 0.0
        total_weight = 0.0
        
        for criterion, score in criteria_scores.items():
            weight = weights.get(criterion, 1.0)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _analyze_strategy_profile(self, criteria_scores: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """Analyze strategy strengths and weaknesses."""
        strengths = []
        weaknesses = []
        
        for criterion, score in criteria_scores.items():
            if score >= 0.8:
                strengths.append(f"Strong {criterion.replace('_', ' ')}")
            elif score <= 0.3:
                weaknesses.append(f"Weak {criterion.replace('_', ' ')}")
        
        return strengths, weaknesses
    
    def _generate_recommendations(self, strategy: Any, criteria_scores: Dict[str, float], context: Dict[str, Any]) -> List[str]:
        """Generate recommendations for strategy improvement."""
        recommendations = []
        
        # Check for specific improvement opportunities
        if criteria_scores.get(EvaluationCriteria.COMPLEXITY.value, 0) < 0.5:
            recommendations.append("Consider simplifying the strategy by reducing steps or dependencies")
        
        if criteria_scores.get(EvaluationCriteria.RISK_MITIGATION.value, 0) < 0.5:
            recommendations.append("Add more fallback options and risk mitigation measures")
        
        if criteria_scores.get(EvaluationCriteria.PARALLEL_POTENTIAL.value, 0) > 0.7:
            recommendations.append("Leverage parallel execution to reduce completion time")
        
        if criteria_scores.get(EvaluationCriteria.SUCCESS_PROBABILITY.value, 0) < 0.6:
            recommendations.append("Increase step confidence levels or add validation checkpoints")
        
        return recommendations
    
    def _calculate_evaluation_confidence(self, strategy: Any, criteria_scores: Dict[str, float], context: Dict[str, Any]) -> float:
        """Calculate confidence in the evaluation."""
        # Base confidence on completeness of information
        base_confidence = 0.7
        
        # Adjust based on historical data availability
        if self._has_historical_data(strategy):
            base_confidence += 0.2
        
        # Adjust based on context completeness
        context_completeness = len(context) / 10  # Assume 10 relevant context factors
        base_confidence += min(0.1, context_completeness * 0.1)
        
        return min(1.0, base_confidence)
    
    def _generate_comparison_summary(self, evaluations: List[EvaluationResult], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of strategy comparison."""
        if not evaluations:
            return {}
        
        best_eval = max(evaluations, key=lambda e: e.overall_score)
        worst_eval = min(evaluations, key=lambda e: e.overall_score)
        
        return {
            "strategy_count": len(evaluations),
            "score_range": {
                "best": best_eval.overall_score,
                "worst": worst_eval.overall_score,
                "average": sum(e.overall_score for e in evaluations) / len(evaluations)
            },
            "top_criteria": self._identify_distinguishing_criteria(evaluations),
            "consensus_strengths": self._find_common_strengths(evaluations),
            "key_differentiators": self._find_key_differentiators(evaluations)
        }
    
    def _generate_decision_rationale(self, evaluations: List[EvaluationResult], rankings: List[Tuple[str, float]], context: Dict[str, Any]) -> str:
        """Generate rationale for the strategy selection decision."""
        if not evaluations or not rankings:
            return "Insufficient data for decision rationale"
        
        best_eval = next(e for e in evaluations if e.strategy_id == rankings[0][0])
        
        rationale_parts = [
            f"Selected strategy {best_eval.strategy_id} with score {best_eval.overall_score:.3f}",
            f"Key strengths: {', '.join(best_eval.strengths[:3])}",
            f"Confidence level: {best_eval.confidence:.1%}"
        ]
        
        if len(rankings) > 1:
            second_best_score = rankings[1][1]
            margin = best_eval.overall_score - second_best_score
            rationale_parts.append(f"Margin over next best option: {margin:.3f}")
        
        return ". ".join(rationale_parts)
    
    def _get_historical_success_rate(self, strategy: Any) -> float:
        """Get historical success rate for similar strategies."""
        # This would analyze historical data
        # For now, return a neutral value
        return 0.5
    
    def _get_strategy_similarity_to_history(self, strategy: Any) -> float:
        """Calculate similarity to historical strategies."""
        # This would compare against historical strategies
        # For now, return a moderate value
        return 0.5
    
    def _has_historical_data(self, strategy: Any) -> bool:
        """Check if historical data is available for this strategy type."""
        return len(self.evaluation_history) > 10
    
    def _identify_distinguishing_criteria(self, evaluations: List[EvaluationResult]) -> List[str]:
        """Identify criteria that best distinguish between strategies."""
        # Calculate variance for each criterion
        criteria_variances = {}
        
        for criterion in EvaluationCriteria:
            scores = [e.criteria_scores.get(criterion.value, 0) for e in evaluations]
            if len(scores) > 1:
                mean = sum(scores) / len(scores)
                variance = sum((s - mean) ** 2 for s in scores) / len(scores)
                criteria_variances[criterion.value] = variance
        
        # Return top 3 criteria with highest variance
        sorted_criteria = sorted(criteria_variances.items(), key=lambda x: x[1], reverse=True)
        return [criterion for criterion, variance in sorted_criteria[:3]]
    
    def _find_common_strengths(self, evaluations: List[EvaluationResult]) -> List[str]:
        """Find strengths common across multiple strategies."""
        strength_counts = {}
        
        for evaluation in evaluations:
            for strength in evaluation.strengths:
                strength_counts[strength] = strength_counts.get(strength, 0) + 1
        
        # Return strengths that appear in at least half the strategies
        threshold = len(evaluations) // 2
        return [strength for strength, count in strength_counts.items() if count >= threshold]
    
    def _find_key_differentiators(self, evaluations: List[EvaluationResult]) -> List[str]:
        """Find key factors that differentiate strategies."""
        # This would analyze the most significant differences
        # For now, return a simple analysis
        return ["Complexity management", "Risk tolerance", "Time constraints"]
    
    def _update_historical_data(self, strategy: Any, evaluation: EvaluationResult) -> None:
        """Update historical evaluation data."""
        strategy_type = strategy.strategy_type.value if hasattr(strategy, 'strategy_type') else 'unknown'
        
        if strategy_type not in self.strategy_history:
            self.strategy_history[strategy_type] = []
        
        self.strategy_history[strategy_type].append({
            'evaluation': evaluation,
            'timestamp': datetime.now()
        })
    
    def _update_weights_from_outcome(self, evaluation: EvaluationResult, outcome: Dict[str, Any]) -> None:
        """Update evaluation weights based on actual outcomes."""
        # This would implement adaptive weight learning
        # For now, just log the intent
        logging.info(f"[{self.name}] Weight adaptation based on outcome (placeholder)")
    
    def _store_outcome_data(self, strategy_id: str, evaluation: EvaluationResult, outcome: Dict[str, Any]) -> None:
        """Store outcome data for future reference."""
        # This would store outcome data for learning
        logging.info(f"[{self.name}] Stored outcome data for strategy {strategy_id}")
    
    def _calculate_average_scores(self) -> Dict[str, float]:
        """Calculate average scores across all evaluations."""
        if not self.evaluation_history:
            return {}
        
        criteria_totals = {}
        criteria_counts = {}
        
        for evaluation in self.evaluation_history:
            for criterion, score in evaluation.criteria_scores.items():
                criteria_totals[criterion] = criteria_totals.get(criterion, 0) + score
                criteria_counts[criterion] = criteria_counts.get(criterion, 0) + 1
        
        return {
            criterion: total / criteria_counts[criterion]
            for criterion, total in criteria_totals.items()
        }
    
    def _analyze_criteria_trends(self) -> Dict[str, Any]:
        """Analyze trends in criteria scores over time."""
        # This would analyze how criteria scores change over time
        return {"trend_analysis": "Not yet implemented"}
    
    def _analyze_context_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in evaluation contexts."""
        # This would analyze common context patterns
        return {"context_patterns": "Not yet implemented"}
    
    def _analyze_weight_adaptations(self) -> Dict[str, Any]:
        """Analyze how weights have been adapted."""
        # This would track weight changes
        return {"weight_changes": "Not yet implemented"}
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate how accurate strategy evaluations have been."""
        # This would compare predictions to actual outcomes
        return 0.0  # Placeholder
    
    def _generate_system_recommendations(self) -> List[str]:
        """Generate recommendations for the evaluation system itself."""
        recommendations = []
        
        if len(self.evaluation_history) < 10:
            recommendations.append("Collect more evaluation data for better insights")
        
        # Add more system-level recommendations
        return recommendations
    
    def _create_default_evaluation(self, strategy_id: str) -> EvaluationResult:
        """Create a default evaluation when the evaluator is disabled."""
        return EvaluationResult(
            strategy_id=strategy_id,
            overall_score=0.5,
            criteria_scores={criterion.value: 0.5 for criterion in EvaluationCriteria},
            strengths=[],
            weaknesses=[],
            recommendations=[],
            confidence=0.3,
            evaluation_timestamp=datetime.now(),
            context_factors={}
        )
    
    def _load_evaluation_config(self) -> Dict[str, Any]:
        """Load evaluation configuration."""
        from config.settings import get_config
        
        config = get_config().get('strategy_evaluator', {})
        
        # Default configuration
        default_config = {
            'enabled': True,
            'confidence_threshold': 0.7,
            'historical_weight': 0.3,
            'context_adaptation_rate': 0.1,
            'base_weights': {
                'feasibility': 1.0,
                'complexity': 0.8,
                'resource_efficiency': 0.9,
                'time_to_completion': 0.7,
                'success_probability': 1.2,
                'risk_mitigation': 0.8,
                'adaptability': 0.6,
                'parallel_potential': 0.5,
                'recovery_options': 0.7,
                'learning_value': 0.4
            }
        }
        
        # Merge with user config
        default_config.update(config)
        return default_config