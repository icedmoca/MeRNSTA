#!/usr/bin/env python3
"""
Agent Lifecycle Manager for MeRNSTA Phase 27

Autonomous agent lifecycle management based on performance, drift, and specialization alignment.
Handles promotion, mutation, and retirement of agents based on configurable criteria.
"""

import logging
import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from .base import BaseAgent
from .agent_contract import AgentContract
from config.settings import get_config


class LifecycleAction(Enum):
    """Types of lifecycle actions."""
    PROMOTE = "promote"
    MUTATE = "mutate"
    RETIRE = "retire"
    MONITOR = "monitor"
    NONE = "none"


@dataclass
class LifecycleMetrics:
    """Metrics for agent lifecycle evaluation."""
    agent_name: str
    drift_score: float = 0.0
    success_rate: float = 0.0
    execution_count: int = 0
    confidence_variance: float = 0.0
    last_evaluation: datetime = field(default_factory=datetime.now)
    performance_trend: List[float] = field(default_factory=list)
    contract_alignment: float = 0.0
    specialization_focus: float = 0.0


@dataclass
class LifecycleDecision:
    """Decision made by the lifecycle manager."""
    agent_name: str
    action: LifecycleAction
    reason: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Optional[LifecycleMetrics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentLifecycleManager(BaseAgent):
    """
    Manages autonomous agent lifecycle based on performance, drift, and alignment.
    
    Capabilities:
    - Evaluate agent drift from their contracts
    - Detect promotion candidates based on high performance
    - Identify agents needing mutation due to drift/underperformance
    - Recommend retirement for misaligned or failing agents
    - Log all lifecycle decisions for transparency
    """
    
    def __init__(self):
        super().__init__("agent_lifecycle")
        
        # Load lifecycle configuration
        self.config = get_config().get('agent_lifecycle', {})
        self.enabled = self.config.get('enabled', True)
        
        # Configurable thresholds
        thresholds = self.config.get('thresholds', {})
        self.drift_threshold = thresholds.get('drift_threshold', 0.7)
        self.promotion_threshold = thresholds.get('promotion_threshold', 0.85)
        self.retirement_threshold = thresholds.get('retirement_threshold', 0.3)
        self.mutation_threshold = thresholds.get('mutation_threshold', 0.5)
        
        # Performance tracking settings
        self.min_execution_count = self.config.get('min_execution_count', 10)
        self.evaluation_window_days = self.config.get('evaluation_window_days', 7)
        self.trend_analysis_depth = self.config.get('trend_analysis_depth', 20)
        
        # Lifecycle action weights (configurable)
        self.action_weights = self.config.get('action_weights', {
            'drift_weight': 0.3,
            'performance_weight': 0.4,
            'alignment_weight': 0.2,
            'trend_weight': 0.1
        })
        
        # Directory setup
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Logging files
        self.lifecycle_log_path = self.output_dir / "lifecycle.log"
        self.lifecycle_jsonl_path = self.output_dir / "lifecycle.jsonl"
        self.metrics_cache_path = self.output_dir / "agent_metrics.json"
        
        # Metrics tracking
        self.agent_metrics: Dict[str, LifecycleMetrics] = {}
        self.decision_history: List[LifecycleDecision] = []
        
        # Load existing metrics
        self._load_existing_metrics()
        
        logging.info(f"[AgentLifecycleManager] Initialized with thresholds: drift={self.drift_threshold}, "
                    f"promotion={self.promotion_threshold}, retirement={self.retirement_threshold}")
    
    def get_agent_instructions(self) -> str:
        """Get instructions for the lifecycle manager."""
        return """You are the Agent Lifecycle Manager for MeRNSTA's autonomous agent management system.

Your primary responsibilities are:
1. Monitor agent performance and drift from their contracts
2. Identify high-performing agents for promotion (confidence boosting)
3. Detect underperforming or drifting agents for mutation
4. Recommend retirement for severely misaligned or failing agents
5. Log all lifecycle decisions with detailed reasoning

Key capabilities:
- Performance-based drift detection using configurable thresholds
- Contract alignment evaluation using vector similarity
- Trend analysis for performance trajectories
- Automated lifecycle action recommendations
- Comprehensive logging and audit trails

Use evidence-based metrics to make lifecycle decisions that optimize
the overall cognitive system performance through strategic agent management."""
    
    def respond(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process lifecycle management requests."""
        
        message_lower = message.lower()
        
        try:
            if any(word in message_lower for word in ['evaluate', 'assess', 'analyze']):
                return self._handle_evaluation_request(message, context)
            
            elif any(word in message_lower for word in ['promote', 'boost', 'upgrade']):
                return self._handle_promotion_request(message, context)
            
            elif any(word in message_lower for word in ['mutate', 'evolve', 'modify']):
                return self._handle_mutation_request(message, context)
            
            elif any(word in message_lower for word in ['retire', 'remove', 'deactivate']):
                return self._handle_retirement_request(message, context)
            
            elif any(word in message_lower for word in ['drift', 'alignment', 'contract']):
                return self._handle_drift_analysis(message, context)
            
            elif any(word in message_lower for word in ['status', 'report', 'summary']):
                return self._handle_status_request(message, context)
            
            else:
                return self._handle_general_inquiry(message, context)
                
        except Exception as e:
            logging.error(f"[AgentLifecycleManager] Error processing message: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to process lifecycle management request',
                'source': 'agent_lifecycle'
            }
    
    def evaluate_drift(self, agent: BaseAgent) -> float:
        """
        Evaluate how much an agent has drifted from its original contract.
        
        Args:
            agent: The agent to evaluate
            
        Returns:
            float: Drift score between 0.0 (no drift) and 1.0 (maximum drift)
        """
        try:
            if not agent.contract:
                logging.warning(f"[AgentLifecycleManager] No contract found for agent {agent.name}")
                return 0.5  # Default moderate drift for agents without contracts
            
            # Get current agent metrics
            current_metrics = agent.get_lifecycle_metrics() if hasattr(agent, 'get_lifecycle_metrics') else {}
            
            # Calculate drift based on confidence vector changes
            contract_confidence = agent.contract.confidence_vector
            current_confidence = current_metrics.get('confidence_vector', {})
            
            if not contract_confidence or not current_confidence:
                return 0.3  # Low drift if we can't compare
            
            # Compute cosine distance between original and current confidence vectors
            drift_score = self._calculate_vector_drift(contract_confidence, current_confidence)
            
            # Factor in performance degradation
            success_rate = current_metrics.get('success_rate', 0.5)
            performance_factor = 1.0 - success_rate
            
            # Combine drift metrics
            combined_drift = (drift_score * 0.7) + (performance_factor * 0.3)
            
            # Ensure result is in [0, 1] range
            final_drift = max(0.0, min(1.0, combined_drift))
            
            logging.debug(f"[AgentLifecycleManager] Agent {agent.name} drift: {final_drift:.3f}")
            return final_drift
            
        except Exception as e:
            logging.error(f"[AgentLifecycleManager] Error evaluating drift for {agent.name}: {e}")
            return 0.5  # Default to moderate drift on error
    
    def should_promote(self, agent: BaseAgent) -> bool:
        """
        Determine if an agent should be promoted based on high performance.
        
        Args:
            agent: The agent to evaluate
            
        Returns:
            bool: True if agent should be promoted
        """
        try:
            metrics = self._get_or_create_metrics(agent.name)
            
            # Check minimum execution threshold
            if metrics.execution_count < self.min_execution_count:
                return False
            
            # Check high success rate
            if metrics.success_rate < self.promotion_threshold:
                return False
            
            # Check low drift (high alignment)
            drift = self.evaluate_drift(agent)
            if drift > (1.0 - self.promotion_threshold):
                return False
            
            # Check positive performance trend
            if len(metrics.performance_trend) >= 3:
                recent_trend = np.mean(metrics.performance_trend[-3:])
                older_trend = np.mean(metrics.performance_trend[-6:-3]) if len(metrics.performance_trend) >= 6 else 0.5
                trend_improvement = recent_trend > older_trend
                
                if not trend_improvement:
                    return False
            
            # Check if recently promoted (avoid over-promotion)
            if hasattr(agent, 'last_promotion'):
                time_since_promotion = datetime.now() - agent.last_promotion
                if time_since_promotion < timedelta(days=self.evaluation_window_days):
                    return False
            
            logging.info(f"[AgentLifecycleManager] Agent {agent.name} qualifies for promotion")
            return True
            
        except Exception as e:
            logging.error(f"[AgentLifecycleManager] Error evaluating promotion for {agent.name}: {e}")
            return False
    
    def should_mutate(self, agent: BaseAgent) -> bool:
        """
        Determine if an agent should be mutated due to drift or underperformance.
        
        Args:
            agent: The agent to evaluate
            
        Returns:
            bool: True if agent should be mutated
        """
        try:
            metrics = self._get_or_create_metrics(agent.name)
            
            # Check minimum execution threshold
            if metrics.execution_count < self.min_execution_count:
                return False
            
            # Check for significant drift
            drift = self.evaluate_drift(agent)
            if drift > self.drift_threshold:
                logging.info(f"[AgentLifecycleManager] Agent {agent.name} has high drift: {drift:.3f}")
                return True
            
            # Check for poor performance
            if metrics.success_rate < self.mutation_threshold:
                logging.info(f"[AgentLifecycleManager] Agent {agent.name} has low success rate: {metrics.success_rate:.3f}")
                return True
            
            # Check for negative performance trend
            if len(metrics.performance_trend) >= 5:
                recent_performance = np.mean(metrics.performance_trend[-3:])
                if recent_performance < self.mutation_threshold:
                    logging.info(f"[AgentLifecycleManager] Agent {agent.name} has declining performance trend")
                    return True
            
            # Check if recently mutated (avoid over-mutation)
            if hasattr(agent, 'last_mutation'):
                time_since_mutation = datetime.now() - agent.last_mutation
                if time_since_mutation < timedelta(days=self.evaluation_window_days / 2):
                    return False
            
            return False
            
        except Exception as e:
            logging.error(f"[AgentLifecycleManager] Error evaluating mutation for {agent.name}: {e}")
            return False
    
    def should_retire(self, agent: BaseAgent) -> bool:
        """
        Determine if an agent should be retired due to severe misalignment or failure.
        
        Args:
            agent: The agent to evaluate
            
        Returns:
            bool: True if agent should be retired
        """
        try:
            metrics = self._get_or_create_metrics(agent.name)
            
            # Need substantial execution history to retire
            if metrics.execution_count < self.min_execution_count * 2:
                return False
            
            # Check for extremely poor performance
            if metrics.success_rate < self.retirement_threshold:
                logging.info(f"[AgentLifecycleManager] Agent {agent.name} has very low success rate: {metrics.success_rate:.3f}")
                return True
            
            # Check for extreme drift
            drift = self.evaluate_drift(agent)
            if drift > 0.9:  # Very high drift threshold for retirement
                logging.info(f"[AgentLifecycleManager] Agent {agent.name} has extreme drift: {drift:.3f}")
                return True
            
            # Check for sustained poor performance
            if len(metrics.performance_trend) >= 10:
                recent_avg = np.mean(metrics.performance_trend[-5:])
                if recent_avg < self.retirement_threshold:
                    logging.info(f"[AgentLifecycleManager] Agent {agent.name} has sustained poor performance")
                    return True
            
            return False
            
        except Exception as e:
            logging.error(f"[AgentLifecycleManager] Error evaluating retirement for {agent.name}: {e}")
            return False
    
    def apply_lifecycle_decision(self, agent: BaseAgent) -> LifecycleDecision:
        """
        Evaluate an agent and apply the appropriate lifecycle decision.
        
        Args:
            agent: The agent to evaluate and potentially act upon
            
        Returns:
            LifecycleDecision: The decision made and action taken
        """
        try:
            # Update metrics first
            self._update_agent_metrics(agent)
            
            # Evaluate lifecycle actions in priority order
            if self.should_retire(agent):
                decision = self._create_retirement_decision(agent)
            elif self.should_promote(agent):
                decision = self._create_promotion_decision(agent)
            elif self.should_mutate(agent):
                decision = self._create_mutation_decision(agent)
            else:
                decision = self._create_monitor_decision(agent)
            
            # Log the decision
            self._log_lifecycle_decision(decision)
            
            # Store in history
            self.decision_history.append(decision)
            
            return decision
            
        except Exception as e:
            logging.error(f"[AgentLifecycleManager] Error applying lifecycle decision for {agent.name}: {e}")
            return LifecycleDecision(
                agent_name=agent.name,
                action=LifecycleAction.NONE,
                reason=f"Error in lifecycle evaluation: {e}",
                confidence=0.0
            )
    
    def _calculate_vector_drift(self, original: Dict[str, float], current: Dict[str, float]) -> float:
        """Calculate cosine distance between two confidence vectors."""
        try:
            # Get common keys
            common_keys = set(original.keys()) & set(current.keys())
            if not common_keys:
                return 1.0  # Maximum drift if no common keys
            
            # Create vectors for common keys
            vec1 = np.array([original[key] for key in common_keys])
            vec2 = np.array([current[key] for key in common_keys])
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            
            if norms == 0:
                return 0.0  # No drift if both vectors are zero
            
            cosine_similarity = dot_product / norms
            cosine_distance = 1.0 - cosine_similarity
            
            return max(0.0, min(1.0, cosine_distance))
            
        except Exception as e:
            logging.error(f"[AgentLifecycleManager] Error calculating vector drift: {e}")
            return 0.5  # Default moderate drift on error
    
    def _get_or_create_metrics(self, agent_name: str) -> LifecycleMetrics:
        """Get or create metrics for an agent."""
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = LifecycleMetrics(agent_name=agent_name)
        return self.agent_metrics[agent_name]
    
    def _update_agent_metrics(self, agent: BaseAgent) -> None:
        """Update stored metrics for an agent."""
        try:
            metrics = self._get_or_create_metrics(agent.name)
            
            # Get current agent metrics if available
            if hasattr(agent, 'get_lifecycle_metrics'):
                current_data = agent.get_lifecycle_metrics()
                
                metrics.success_rate = current_data.get('success_rate', metrics.success_rate)
                metrics.execution_count = current_data.get('execution_count', metrics.execution_count)
                
                # Update performance trend
                if 'current_performance' in current_data:
                    metrics.performance_trend.append(current_data['current_performance'])
                    if len(metrics.performance_trend) > self.trend_analysis_depth:
                        metrics.performance_trend = metrics.performance_trend[-self.trend_analysis_depth:]
            
            # Calculate current drift
            metrics.drift_score = self.evaluate_drift(agent)
            
            # Update contract alignment
            if agent.contract:
                metrics.contract_alignment = 1.0 - metrics.drift_score
            
            metrics.last_evaluation = datetime.now()
            
            # Save metrics to cache
            self._save_metrics_cache()
            
        except Exception as e:
            logging.error(f"[AgentLifecycleManager] Error updating metrics for {agent.name}: {e}")
    
    def _create_promotion_decision(self, agent: BaseAgent) -> LifecycleDecision:
        """Create a promotion decision for an agent."""
        metrics = self._get_or_create_metrics(agent.name)
        reason = f"High performance: success_rate={metrics.success_rate:.3f}, low_drift={1.0-metrics.drift_score:.3f}"
        
        return LifecycleDecision(
            agent_name=agent.name,
            action=LifecycleAction.PROMOTE,
            reason=reason,
            confidence=0.9,
            metrics=metrics
        )
    
    def _create_mutation_decision(self, agent: BaseAgent) -> LifecycleDecision:
        """Create a mutation decision for an agent."""
        metrics = self._get_or_create_metrics(agent.name)
        reason = f"Performance/drift issues: success_rate={metrics.success_rate:.3f}, drift={metrics.drift_score:.3f}"
        
        return LifecycleDecision(
            agent_name=agent.name,
            action=LifecycleAction.MUTATE,
            reason=reason,
            confidence=0.8,
            metrics=metrics
        )
    
    def _create_retirement_decision(self, agent: BaseAgent) -> LifecycleDecision:
        """Create a retirement decision for an agent."""
        metrics = self._get_or_create_metrics(agent.name)
        reason = f"Severe performance/alignment issues: success_rate={metrics.success_rate:.3f}, drift={metrics.drift_score:.3f}"
        
        return LifecycleDecision(
            agent_name=agent.name,
            action=LifecycleAction.RETIRE,
            reason=reason,
            confidence=0.95,
            metrics=metrics
        )
    
    def _create_monitor_decision(self, agent: BaseAgent) -> LifecycleDecision:
        """Create a monitoring decision for an agent."""
        metrics = self._get_or_create_metrics(agent.name)
        reason = "Performance within acceptable ranges, continue monitoring"
        
        return LifecycleDecision(
            agent_name=agent.name,
            action=LifecycleAction.MONITOR,
            reason=reason,
            confidence=0.7,
            metrics=metrics
        )
    
    def _log_lifecycle_decision(self, decision: LifecycleDecision) -> None:
        """Log a lifecycle decision to both text and JSON logs."""
        try:
            # Text log
            log_message = (f"[{decision.timestamp}] LIFECYCLE: {decision.agent_name} -> "
                          f"{decision.action.value.upper()} (confidence: {decision.confidence:.3f}) "
                          f"Reason: {decision.reason}")
            
            with open(self.lifecycle_log_path, 'a') as f:
                f.write(log_message + '\n')
            
            # JSON log
            decision_data = {
                'timestamp': decision.timestamp.isoformat(),
                'agent_name': decision.agent_name,
                'action': decision.action.value,
                'reason': decision.reason,
                'confidence': decision.confidence,
                'metrics': {
                    'drift_score': decision.metrics.drift_score if decision.metrics else 0.0,
                    'success_rate': decision.metrics.success_rate if decision.metrics else 0.0,
                    'execution_count': decision.metrics.execution_count if decision.metrics else 0
                } if decision.metrics else {}
            }
            
            with open(self.lifecycle_jsonl_path, 'a') as f:
                f.write(json.dumps(decision_data) + '\n')
            
            logging.info(log_message)
            
        except Exception as e:
            logging.error(f"[AgentLifecycleManager] Error logging decision: {e}")
    
    def _load_existing_metrics(self) -> None:
        """Load existing metrics from cache."""
        try:
            if self.metrics_cache_path.exists():
                with open(self.metrics_cache_path, 'r') as f:
                    data = json.load(f)
                    
                for agent_name, metrics_data in data.items():
                    metrics = LifecycleMetrics(agent_name=agent_name)
                    metrics.drift_score = metrics_data.get('drift_score', 0.0)
                    metrics.success_rate = metrics_data.get('success_rate', 0.0)
                    metrics.execution_count = metrics_data.get('execution_count', 0)
                    metrics.performance_trend = metrics_data.get('performance_trend', [])
                    
                    self.agent_metrics[agent_name] = metrics
                    
                logging.info(f"[AgentLifecycleManager] Loaded metrics for {len(self.agent_metrics)} agents")
                
        except Exception as e:
            logging.error(f"[AgentLifecycleManager] Error loading existing metrics: {e}")
    
    def _save_metrics_cache(self) -> None:
        """Save current metrics to cache."""
        try:
            data = {}
            for agent_name, metrics in self.agent_metrics.items():
                data[agent_name] = {
                    'drift_score': metrics.drift_score,
                    'success_rate': metrics.success_rate,
                    'execution_count': metrics.execution_count,
                    'performance_trend': metrics.performance_trend,
                    'last_evaluation': metrics.last_evaluation.isoformat()
                }
            
            with open(self.metrics_cache_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logging.error(f"[AgentLifecycleManager] Error saving metrics cache: {e}")
    
    def _handle_evaluation_request(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle requests to evaluate agents."""
        return {
            'success': True,
            'message': 'Lifecycle evaluation capabilities ready',
            'available_evaluations': ['drift', 'promotion', 'mutation', 'retirement'],
            'source': 'agent_lifecycle'
        }
    
    def _handle_promotion_request(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle promotion requests."""
        return {
            'success': True,
            'message': 'Promotion evaluation system ready',
            'criteria': f'Success rate > {self.promotion_threshold}, low drift, positive trends',
            'source': 'agent_lifecycle'
        }
    
    def _handle_mutation_request(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle mutation requests."""
        return {
            'success': True,
            'message': 'Mutation evaluation system ready',
            'criteria': f'High drift > {self.drift_threshold} or low performance < {self.mutation_threshold}',
            'source': 'agent_lifecycle'
        }
    
    def _handle_retirement_request(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle retirement requests."""
        return {
            'success': True,
            'message': 'Retirement evaluation system ready',
            'criteria': f'Very low performance < {self.retirement_threshold} or extreme drift > 0.9',
            'source': 'agent_lifecycle'
        }
    
    def _handle_drift_analysis(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle drift analysis requests."""
        return {
            'success': True,
            'message': 'Drift analysis capabilities ready',
            'method': 'Cosine distance between original and current confidence vectors',
            'source': 'agent_lifecycle'
        }
    
    def _handle_status_request(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle status requests."""
        return {
            'success': True,
            'message': 'Lifecycle manager operational',
            'tracked_agents': len(self.agent_metrics),
            'decisions_made': len(self.decision_history),
            'thresholds': {
                'drift': self.drift_threshold,
                'promotion': self.promotion_threshold,
                'mutation': self.mutation_threshold,
                'retirement': self.retirement_threshold
            },
            'source': 'agent_lifecycle'
        }
    
    def _handle_general_inquiry(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle general inquiries about lifecycle management."""
        return {
            'success': True,
            'message': 'Agent Lifecycle Manager: Autonomous agent performance monitoring and lifecycle management',
            'capabilities': [
                'Performance drift detection',
                'Promotion recommendation',
                'Mutation triggering',
                'Retirement decisions',
                'Contract alignment monitoring'
            ],
            'source': 'agent_lifecycle'
        }