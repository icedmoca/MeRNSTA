#!/usr/bin/env python3
"""
Meta Router - Introspective Subgoal Routing

This module implements intelligent routing of subgoals to specialized agents
based on symbolic reasoning and repair simulation results.
"""

import time
import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
import numpy as np

from storage.reflex_log import ReflexLogger, get_reflex_logger
from storage.self_model import CognitiveSelfModel
from agents.repair_simulator import RepairSimulator, get_repair_simulator, SimulatedRepair
from .agent_contract import score_all_agents_for_task


@dataclass
class RoutedSubgoal:
    """Represents a subgoal that has been routed to a specific agent."""
    subgoal_id: str
    goal_id: str
    agent_type: str  # reflector, clarifier, consolidator, etc.
    priority: float
    reasoning_path: List[str]  # Traceable decision path
    estimated_duration: float
    confidence: float
    dependencies: List[str]  # Other subgoals this depends on
    tags: List[str]  # Goal tags for categorization
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RoutedSubgoal':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class RoutingPlan:
    """Complete routing plan for a goal with multiple subgoals."""
    goal_id: str
    subgoals: List[RoutedSubgoal]
    total_estimated_duration: float
    routing_confidence: float
    reasoning_summary: str
    routing_timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['subgoals'] = [s.to_dict() for s in self.subgoals]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RoutingPlan':
        """Create from dictionary."""
        subgoals_data = data.pop('subgoals', [])
        
        result = cls(**data)
        result.subgoals = [RoutedSubgoal.from_dict(s) for s in subgoals_data]
        return result


class MetaRouter:
    """
    Intelligent router that assigns subgoals to specialized agents.
    
    Features:
    - Uses repair simulation results for routing decisions
    - Applies symbolic reasoning from self-model
    - Logs traceable decision paths
    - Supports goal tagging and dependency tracking
    - Optimizes routing based on agent capabilities
    """
    
    def __init__(self, reflex_logger: Optional[ReflexLogger] = None,
                 self_model: Optional[CognitiveSelfModel] = None,
                 repair_simulator: Optional[RepairSimulator] = None):
        self.reflex_logger = reflex_logger or get_reflex_logger()
        self.self_model = self_model or CognitiveSelfModel()
        self.repair_simulator = repair_simulator or get_repair_simulator()
        
        # Available agent types and their capabilities
        self.agent_capabilities = {
            'reflector': {
                'strategies': ['belief_clarification'],
                'strengths': ['introspection', 'belief_analysis', 'contradiction_resolution'],
                'max_duration': 120.0,
                'priority_weight': 0.8
            },
            'clarifier': {
                'strategies': ['belief_clarification', 'fact_consolidation'],
                'strengths': ['fact_verification', 'ambiguity_resolution', 'context_clarification'],
                'max_duration': 90.0,
                'priority_weight': 0.7
            },
            'consolidator': {
                'strategies': ['fact_consolidation', 'cluster_reassessment'],
                'strengths': ['memory_integration', 'pattern_recognition', 'coherence_improvement'],
                'max_duration': 150.0,
                'priority_weight': 0.9
            },
            'anticipator': {
                'strategies': ['anticipatory_drift'],
                'strengths': ['drift_prediction', 'proactive_repair', 'trend_analysis'],
                'max_duration': 60.0,
                'priority_weight': 0.6
            },
            'optimizer': {
                'strategies': ['belief_clarification', 'cluster_reassessment', 'fact_consolidation'],
                'strengths': ['strategy_optimization', 'performance_analysis', 'adaptive_routing'],
                'max_duration': 180.0,
                'priority_weight': 1.0
            }
        }
        
        # Routing parameters
        self.min_confidence_threshold = 0.4
        self.max_subgoals_per_goal = 5
        self.routing_weights = {
            'simulation_score': 0.4,
            'agent_capability': 0.3,
            'symbolic_reasoning': 0.2,
            'load_balancing': 0.1
        }
        
        print(f"[MetaRouter] Initialized meta router")
    
    def route_subgoals(self, goal_id: str, current_state: Dict[str, Any], 
                      goal_description: str = "", tags: List[str] = None) -> RoutingPlan:
        """
        Route subgoals for a given goal using intelligent decision making.
        
        Args:
            goal_id: Identifier for the goal
            current_state: Current cognitive state
            goal_description: Description of the goal
            tags: Optional tags for the goal
            
        Returns:
            RoutingPlan with routed subgoals
        """
        try:
            tags = tags or []
            
            # Simulate repair strategies for the goal
            simulation_result = self.repair_simulator.simulate_repair_paths(
                goal_id, current_state
            )
            
            # Generate subgoals based on simulation results
            subgoals = self._generate_subgoals_from_simulation(
                goal_id, simulation_result, current_state, tags
            )
            
            # Route each subgoal to appropriate agents
            routed_subgoals = []
            for subgoal in subgoals:
                routed_subgoal = self._route_single_subgoal(
                    subgoal, current_state, simulation_result
                )
                if routed_subgoal:
                    routed_subgoals.append(routed_subgoal)
            
            # Optimize routing plan
            optimized_subgoals = self._optimize_routing_plan(routed_subgoals)
            
            # Calculate plan metrics
            total_duration = sum(s.estimated_duration for s in optimized_subgoals)
            routing_confidence = np.mean([s.confidence for s in optimized_subgoals]) if optimized_subgoals else 0.0
            
            # Generate reasoning summary
            reasoning_summary = self._generate_routing_summary(
                optimized_subgoals, simulation_result, goal_description
            )
            
            # Log routing decision
            self._log_routing_decision(goal_id, optimized_subgoals, reasoning_summary)
            
            return RoutingPlan(
                goal_id=goal_id,
                subgoals=optimized_subgoals,
                total_estimated_duration=total_duration,
                routing_confidence=routing_confidence,
                reasoning_summary=reasoning_summary,
                routing_timestamp=time.time()
            )
            
        except Exception as e:
            logging.error(f"[MetaRouter] Error routing subgoals: {e}")
            return RoutingPlan(
                goal_id=goal_id,
                subgoals=[],
                total_estimated_duration=0.0,
                routing_confidence=0.0,
                reasoning_summary=f"Routing failed: {e}",
                routing_timestamp=time.time()
            )
    
    def route_task_with_contracts(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Route a task to the best agent using contract-based alignment scoring.
        
        This method provides an alternative to the traditional subgoal routing
        by using agent contracts to determine the best fit for a given task.
        
        Args:
            task: Task description or goal to route
            context: Additional context for the task (urgency, complexity, etc.)
            
        Returns:
            Dictionary with routing results including best agent and justification
        """
        try:
            # Score all agents using their contracts
            agent_scores = score_all_agents_for_task(task)
            
            if not agent_scores:
                logging.warning(f"[MetaRouter] No agents available for contract-based routing")
                return {
                    'success': False,
                    'error': 'No agents available',
                    'task': task,
                    'agents_scored': 0
                }
            
            # Get the top agent
            best_agent = agent_scores[0]
            alternatives = agent_scores[1:5]  # Top 5 alternatives
            
            # Generate routing justification
            justification = self._generate_contract_routing_justification(
                task, best_agent, alternatives, context
            )
            
            # Log the routing decision
            logging.info(f"[MetaRouter] Contract-based routing for task '{task[:50]}...'")
            logging.info(f"[MetaRouter] Selected: {best_agent['agent_name']} "
                        f"(score: {best_agent['alignment_score']:.3f})")
            
            # Check if the top score is reasonable
            min_score_threshold = 0.3
            if best_agent['alignment_score'] < min_score_threshold:
                logging.warning(f"[MetaRouter] Best agent score ({best_agent['alignment_score']:.3f}) "
                               f"below threshold ({min_score_threshold})")
            
            return {
                'success': True,
                'task': task,
                'selected_agent': best_agent,
                'alternatives': alternatives,
                'justification': justification,
                'agents_scored': len(agent_scores),
                'routing_confidence': best_agent['alignment_score'],
                'routing_timestamp': time.time(),
                'context': context
            }
            
        except Exception as e:
            logging.error(f"[MetaRouter] Error in contract-based routing: {e}")
            return {
                'success': False,
                'error': str(e),
                'task': task,
                'agents_scored': 0
            }
    
    def _generate_contract_routing_justification(self, task: str, best_agent: Dict[str, Any], 
                                               alternatives: List[Dict[str, Any]], 
                                               context: Dict[str, Any] = None) -> str:
        """Generate human-readable justification for contract-based routing decision."""
        
        justification_parts = [
            f"Task: '{task[:100]}{'...' if len(task) > 100 else ''}'",
            f"",
            f"🎯 Selected Agent: {best_agent['agent_name']} (Score: {best_agent['alignment_score']:.3f})",
            f"Purpose: {best_agent['purpose']}"
        ]
        
        # Add top capabilities
        if 'confidence_vector' in best_agent and best_agent['confidence_vector']:
            top_capabilities = sorted(best_agent['confidence_vector'].items(), 
                                    key=lambda x: x[1], reverse=True)[:3]
            justification_parts.append(f"Top capabilities: {', '.join(f'{cap}({conf:.2f})' for cap, conf in top_capabilities)}")
        
        justification_parts.append("")
        
        # Add context factors if available
        if context:
            justification_parts.append("Context factors:")
            for key, value in context.items():
                justification_parts.append(f"  - {key}: {value}")
            justification_parts.append("")
        
        # Add alternatives
        if alternatives:
            justification_parts.append("Alternative agents considered:")
            for i, alt in enumerate(alternatives[:3], 1):
                justification_parts.append(f"  {i}. {alt['agent_name']} (Score: {alt['alignment_score']:.3f})")
        
        # Add score interpretation
        score = best_agent['alignment_score']
        if score > 0.8:
            interpretation = "Excellent fit - agent is highly specialized for this task"
        elif score > 0.6:
            interpretation = "Good fit - agent has strong alignment with task requirements"
        elif score > 0.4:
            interpretation = "Moderate fit - agent can handle the task but may not be optimal"
        else:
            interpretation = "Weak fit - consider if this is the best available option"
        
        justification_parts.extend([
            "",
            f"Score interpretation: {interpretation}",
            f"Routing confidence: {'High' if score > 0.7 else 'Medium' if score > 0.4 else 'Low'}"
        ])
        
        return "\n".join(justification_parts)
    
    def route_with_hybrid_approach(self, goal_id: str, task_description: str, 
                                 current_state: Dict[str, Any], 
                                 use_contracts: bool = True) -> Dict[str, Any]:
        """
        Route using both traditional subgoal routing and contract-based routing.
        
        Args:
            goal_id: Goal identifier
            task_description: Description of the task/goal
            current_state: Current cognitive state
            use_contracts: Whether to use contract-based routing as primary method
            
        Returns:
            Combined routing result with both approaches
        """
        try:
            results = {
                'goal_id': goal_id,
                'task_description': task_description,
                'hybrid_routing': True,
                'timestamp': time.time()
            }
            
            # Contract-based routing
            if use_contracts:
                contract_result = self.route_task_with_contracts(task_description, current_state)
                results['contract_routing'] = contract_result
            
            # Traditional subgoal routing
            traditional_result = self.route_subgoals(goal_id, current_state, task_description)
            results['traditional_routing'] = {
                'success': True,
                'subgoals_count': len(traditional_result.subgoals),
                'total_duration': traditional_result.total_estimated_duration,
                'confidence': traditional_result.routing_confidence,
                'subgoals': [
                    {
                        'id': sg.subgoal_id,
                        'agent': sg.agent_type,
                        'confidence': sg.confidence,
                        'duration': sg.estimated_duration
                    } 
                    for sg in traditional_result.subgoals
                ]
            }
            
            # Determine primary recommendation
            if use_contracts and contract_result['success']:
                results['primary_recommendation'] = 'contract_based'
                results['recommended_agent'] = contract_result['selected_agent']['agent_name']
                results['recommendation_confidence'] = contract_result['routing_confidence']
            elif traditional_result.subgoals:
                results['primary_recommendation'] = 'traditional_subgoals'
                results['recommended_agent'] = traditional_result.subgoals[0].agent_type
                results['recommendation_confidence'] = traditional_result.routing_confidence
            else:
                results['primary_recommendation'] = 'none'
                results['recommended_agent'] = None
                results['recommendation_confidence'] = 0.0
            
            return results
            
        except Exception as e:
            logging.error(f"[MetaRouter] Error in hybrid routing: {e}")
            return {
                'goal_id': goal_id,
                'task_description': task_description,
                'hybrid_routing': True,
                'error': str(e),
                'timestamp': time.time()
            }

    def _generate_subgoals_from_simulation(self, goal_id: str, 
                                         simulation_result: 'RepairSimulationResult',
                                         current_state: Dict[str, Any],
                                         tags: List[str]) -> List[Dict[str, Any]]:
        """Generate subgoals from repair simulation results."""
        subgoals = []
        
        try:
            # Create subgoals for each simulated repair strategy
            for i, repair in enumerate(simulation_result.simulated_repairs):
                if repair.predicted_score > 0.3:  # Minimum threshold
                    subgoal = {
                        'subgoal_id': f"{goal_id}_subgoal_{i+1}",
                        'goal_id': goal_id,
                        'strategy': repair.strategy,
                        'predicted_score': repair.predicted_score,
                        'confidence': repair.confidence,
                        'estimated_duration': repair.estimated_duration,
                        'risk_factors': repair.risk_factors,
                        'reasoning_chain': repair.reasoning_chain,
                        'priority': repair.predicted_score * repair.confidence,
                        'tags': tags + [f"strategy:{repair.strategy}"]
                    }
                    subgoals.append(subgoal)
            
            # Add specialized subgoals based on current state
            specialized_subgoals = self._generate_specialized_subgoals(
                goal_id, current_state, tags
            )
            subgoals.extend(specialized_subgoals)
            
            # Sort by priority
            subgoals.sort(key=lambda x: x['priority'], reverse=True)
            
            # Limit number of subgoals
            subgoals = subgoals[:self.max_subgoals_per_goal]
            
        except Exception as e:
            logging.error(f"[MetaRouter] Error generating subgoals: {e}")
        
        return subgoals
    
    def _generate_specialized_subgoals(self, goal_id: str, current_state: Dict[str, Any],
                                     tags: List[str]) -> List[Dict[str, Any]]:
        """Generate specialized subgoals based on current state analysis."""
        specialized_subgoals = []
        
        try:
            # Contradiction resolution subgoal
            contradiction_count = current_state.get('contradiction_count', 0)
            if contradiction_count > 2:
                specialized_subgoals.append({
                    'subgoal_id': f"{goal_id}_contradiction_resolution",
                    'goal_id': goal_id,
                    'strategy': 'belief_clarification',
                    'predicted_score': 0.7,
                    'confidence': 0.8,
                    'estimated_duration': 45.0,
                    'risk_factors': [f"High contradiction count: {contradiction_count}"],
                    'reasoning_chain': [f"Contradiction count {contradiction_count} requires resolution"],
                    'priority': 0.56,
                    'tags': tags + ['contradiction_resolution']
                })
            
            # Volatility stabilization subgoal
            volatility_score = current_state.get('volatility_score', 0.0)
            if volatility_score > 0.6:
                specialized_subgoals.append({
                    'subgoal_id': f"{goal_id}_volatility_stabilization",
                    'goal_id': goal_id,
                    'strategy': 'cluster_reassessment',
                    'predicted_score': 0.6,
                    'confidence': 0.7,
                    'estimated_duration': 60.0,
                    'risk_factors': [f"High volatility: {volatility_score:.2f}"],
                    'reasoning_chain': [f"Volatility {volatility_score:.2f} requires stabilization"],
                    'priority': 0.42,
                    'tags': tags + ['volatility_stabilization']
                })
            
            # Coherence improvement subgoal
            coherence_score = current_state.get('coherence_score', 1.0)
            if coherence_score < 0.5:
                specialized_subgoals.append({
                    'subgoal_id': f"{goal_id}_coherence_improvement",
                    'goal_id': goal_id,
                    'strategy': 'fact_consolidation',
                    'predicted_score': 0.65,
                    'confidence': 0.75,
                    'estimated_duration': 75.0,
                    'risk_factors': [f"Low coherence: {coherence_score:.2f}"],
                    'reasoning_chain': [f"Coherence {coherence_score:.2f} requires improvement"],
                    'priority': 0.49,
                    'tags': tags + ['coherence_improvement']
                })
        
        except Exception as e:
            logging.error(f"[MetaRouter] Error generating specialized subgoals: {e}")
        
        return specialized_subgoals
    
    def _route_single_subgoal(self, subgoal: Dict[str, Any], 
                            current_state: Dict[str, Any],
                            simulation_result: 'RepairSimulationResult') -> Optional[RoutedSubgoal]:
        """Route a single subgoal to the most appropriate agent."""
        try:
            strategy = subgoal['strategy']
            
            # Find compatible agents
            compatible_agents = []
            for agent_type, capabilities in self.agent_capabilities.items():
                if strategy in capabilities['strategies']:
                    compatible_agents.append((agent_type, capabilities))
            
            if not compatible_agents:
                logging.warning(f"[MetaRouter] No compatible agents for strategy: {strategy}")
                return None
            
            # Score each compatible agent
            agent_scores = []
            for agent_type, capabilities in compatible_agents:
                score = self._calculate_agent_score(
                    agent_type, capabilities, subgoal, current_state, simulation_result
                )
                agent_scores.append((agent_type, capabilities, score))
            
            # Select best agent
            agent_scores.sort(key=lambda x: x[2], reverse=True)
            best_agent_type, best_capabilities, best_score = agent_scores[0]
            
            # Generate reasoning path
            reasoning_path = self._generate_routing_reasoning(
                subgoal, best_agent_type, best_capabilities, agent_scores
            )
            
            # Calculate routing confidence
            confidence = self._calculate_routing_confidence(
                best_score, subgoal, best_capabilities
            )
            
            # Determine dependencies
            dependencies = self._determine_subgoal_dependencies(subgoal, simulation_result)
            
            return RoutedSubgoal(
                subgoal_id=subgoal['subgoal_id'],
                goal_id=subgoal['goal_id'],
                agent_type=best_agent_type,
                priority=subgoal['priority'],
                reasoning_path=reasoning_path,
                estimated_duration=min(subgoal['estimated_duration'], best_capabilities['max_duration']),
                confidence=confidence,
                dependencies=dependencies,
                tags=subgoal['tags']
            )
            
        except Exception as e:
            logging.error(f"[MetaRouter] Error routing subgoal {subgoal.get('subgoal_id', 'unknown')}: {e}")
            return None
    
    def _calculate_agent_score(self, agent_type: str, capabilities: Dict[str, Any],
                             subgoal: Dict[str, Any], current_state: Dict[str, Any],
                             simulation_result: 'RepairSimulationResult') -> float:
        """Calculate how well an agent matches a subgoal."""
        score = 0.0
        
        try:
            # Strategy compatibility
            if subgoal['strategy'] in capabilities['strengths']:
                score += 0.3
            
            # Capability strength matching
            for strength in capabilities['strengths']:
                if strength in subgoal.get('tags', []):
                    score += 0.2
                if strength in str(subgoal.get('reasoning_chain', [])):
                    score += 0.1
            
            # Duration compatibility
            duration_ratio = subgoal['estimated_duration'] / capabilities['max_duration']
            if duration_ratio <= 1.0:
                score += 0.2 * (1.0 - duration_ratio)
            
            # Priority weight
            score += 0.1 * capabilities['priority_weight']
            
            # Symbolic reasoning bonus
            symbolic_bonus = self._get_symbolic_routing_bonus(agent_type, subgoal, current_state)
            score += 0.1 * symbolic_bonus
            
            # Load balancing consideration
            load_penalty = self._get_load_balancing_penalty(agent_type)
            score -= 0.1 * load_penalty
            
        except Exception as e:
            logging.warning(f"[MetaRouter] Error calculating agent score: {e}")
        
        return max(0.0, min(1.0, score))
    
    def _get_symbolic_routing_bonus(self, agent_type: str, subgoal: Dict[str, Any],
                                  current_state: Dict[str, Any]) -> float:
        """Get symbolic reasoning bonus for agent routing."""
        try:
            # Query symbolic rules for agent type
            rules = self.self_model.query_rules(f"Agent({agent_type})")
            
            bonus = 0.0
            for rule in rules:
                if rule.confidence > self.min_confidence_threshold:
                    # Check if rule applies to current subgoal
                    if any(tag in rule.antecedent for tag in subgoal.get('tags', [])):
                        bonus += 0.1 * rule.confidence
                    
                    # Check if rule applies to current state
                    drift_type = current_state.get('drift_type', 'unknown')
                    if drift_type in rule.antecedent:
                        bonus += 0.1 * rule.confidence
            
            return min(1.0, bonus)
            
        except Exception as e:
            logging.warning(f"[MetaRouter] Error getting symbolic routing bonus: {e}")
            return 0.0
    
    def _get_load_balancing_penalty(self, agent_type: str) -> float:
        """Get load balancing penalty for agent type."""
        # This would ideally track current agent load
        # For now, return a small random penalty
        import random
        return random.uniform(0.0, 0.2)
    
    def _generate_routing_reasoning(self, subgoal: Dict[str, Any], agent_type: str,
                                  capabilities: Dict[str, Any],
                                  agent_scores: List[Tuple[str, Dict[str, Any], float]]) -> List[str]:
        """Generate reasoning path for routing decision."""
        reasoning = []
        
        reasoning.append(f"Subgoal: {subgoal['subgoal_id']}")
        reasoning.append(f"Strategy: {subgoal['strategy']}")
        reasoning.append(f"Selected agent: {agent_type}")
        reasoning.append(f"Agent score: {agent_scores[0][2]:.3f}")
        
        # Explain why this agent was chosen
        reasoning.append("Selection reasoning:")
        reasoning.append(f"  - Compatible with strategy: {subgoal['strategy']}")
        reasoning.append(f"  - Strengths: {', '.join(capabilities['strengths'])}")
        reasoning.append(f"  - Max duration: {capabilities['max_duration']}s")
        
        # Compare with other agents
        if len(agent_scores) > 1:
            reasoning.append("Alternative agents:")
            for i, (alt_agent, alt_cap, alt_score) in enumerate(agent_scores[1:3], 1):
                reasoning.append(f"  {i}. {alt_agent} (score: {alt_score:.3f})")
        
        return reasoning
    
    def _calculate_routing_confidence(self, agent_score: float, subgoal: Dict[str, Any],
                                    capabilities: Dict[str, Any]) -> float:
        """Calculate confidence in the routing decision."""
        confidence = agent_score * 0.6  # Base confidence from agent score
        
        # Subgoal confidence contribution
        confidence += subgoal['confidence'] * 0.3
        
        # Capability strength contribution
        capability_strength = len(capabilities['strengths']) / 5.0  # Normalize
        confidence += capability_strength * 0.1
        
        return min(1.0, confidence)
    
    def _determine_subgoal_dependencies(self, subgoal: Dict[str, Any],
                                      simulation_result: 'RepairSimulationResult') -> List[str]:
        """Determine dependencies between subgoals."""
        dependencies = []
        
        try:
            # Check for logical dependencies based on strategy
            strategy = subgoal['strategy']
            
            if strategy == 'fact_consolidation':
                # May depend on belief clarification
                for repair in simulation_result.simulated_repairs:
                    if repair.strategy == 'belief_clarification':
                        dependencies.append(f"{subgoal['goal_id']}_belief_clarification")
            
            elif strategy == 'cluster_reassessment':
                # May depend on fact consolidation
                for repair in simulation_result.simulated_repairs:
                    if repair.strategy == 'fact_consolidation':
                        dependencies.append(f"{subgoal['goal_id']}_fact_consolidation")
            
            # Check for contradiction resolution dependencies
            if 'contradiction_resolution' in subgoal.get('tags', []):
                # Contradiction resolution should happen before other repairs
                pass  # This would be a prerequisite for other subgoals
        
        except Exception as e:
            logging.warning(f"[MetaRouter] Error determining dependencies: {e}")
        
        return dependencies
    
    def _optimize_routing_plan(self, routed_subgoals: List[RoutedSubgoal]) -> List[RoutedSubgoal]:
        """Optimize the routing plan for efficiency and effectiveness."""
        try:
            if not routed_subgoals:
                return []
            
            # Sort by priority and confidence
            optimized = sorted(routed_subgoals, 
                             key=lambda x: (x.priority * x.confidence), reverse=True)
            
            # Remove redundant subgoals (same agent type with similar strategies)
            deduplicated = []
            agent_types_used = set()
            
            for subgoal in optimized:
                if subgoal.agent_type not in agent_types_used:
                    deduplicated.append(subgoal)
                    agent_types_used.add(subgoal.agent_type)
                else:
                    # Check if this subgoal is significantly different
                    existing = next(s for s in deduplicated if s.agent_type == subgoal.agent_type)
                    if self._subgoals_are_different(existing, subgoal):
                        deduplicated.append(subgoal)
            
            return deduplicated
            
        except Exception as e:
            logging.error(f"[MetaRouter] Error optimizing routing plan: {e}")
            return routed_subgoals
    
    def _subgoals_are_different(self, subgoal1: RoutedSubgoal, subgoal2: RoutedSubgoal) -> bool:
        """Check if two subgoals are significantly different."""
        # Check tags for differences
        tags1 = set(subgoal1.tags)
        tags2 = set(subgoal2.tags)
        
        # If they have different specialized tags, they're different
        specialized_tags1 = {t for t in tags1 if ':' in t}
        specialized_tags2 = {t for t in tags2 if ':' in t}
        
        if specialized_tags1 != specialized_tags2:
            return True
        
        # Check priority difference
        priority_diff = abs(subgoal1.priority - subgoal2.priority)
        if priority_diff > 0.2:
            return True
        
        return False
    
    def _generate_routing_summary(self, subgoals: List[RoutedSubgoal],
                                simulation_result: 'RepairSimulationResult',
                                goal_description: str) -> str:
        """Generate a summary of the routing decisions."""
        if not subgoals:
            return "No subgoals were routed."
        
        summary = f"Routed {len(subgoals)} subgoals:\n"
        
        # Group by agent type
        agent_groups = {}
        for subgoal in subgoals:
            if subgoal.agent_type not in agent_groups:
                agent_groups[subgoal.agent_type] = []
            agent_groups[subgoal.agent_type].append(subgoal)
        
        for agent_type, agent_subgoals in agent_groups.items():
            summary += f"\n{agent_type.title()} ({len(agent_subgoals)} subgoals):"
            for subgoal in agent_subgoals:
                summary += f"\n  - {subgoal.subgoal_id} (priority: {subgoal.priority:.2f}, confidence: {subgoal.confidence:.2f})"
        
        # Add simulation context
        if simulation_result.best_repair:
            summary += f"\n\nBest simulated strategy: {simulation_result.best_repair.strategy}"
            summary += f"\nPredicted score: {simulation_result.best_repair.predicted_score:.3f}"
        
        return summary
    
    def _log_routing_decision(self, goal_id: str, subgoals: List[RoutedSubgoal],
                            reasoning_summary: str):
        """Log the routing decision for traceability."""
        try:
            # Create a reflex cycle entry for the routing decision
            routing_cycle = {
                'cycle_id': f"routing_{goal_id}_{int(time.time())}",
                'goal_id': goal_id,
                'strategy': 'meta_routing',
                'actions_taken': [f"Routed {len(subgoals)} subgoals"],
                'completion_notes': reasoning_summary,
                'success': len(subgoals) > 0,
                'timestamp': time.time()
            }
            
            # Log to reflex log
            # Note: This would need to be adapted to the actual reflex logging format
            logging.info(f"[MetaRouter] Routing decision logged: {routing_cycle['cycle_id']}")
            
        except Exception as e:
            logging.error(f"[MetaRouter] Error logging routing decision: {e}")


# Global meta router instance
_meta_router_instance = None


def get_meta_router() -> MetaRouter:
    """Get or create the global meta router instance."""
    global _meta_router_instance
    
    if _meta_router_instance is None:
        _meta_router_instance = MetaRouter()
    
    return _meta_router_instance


def route_subgoals(goal_id: str, current_state: Dict[str, Any], 
                  goal_description: str = "", tags: List[str] = None) -> RoutingPlan:
    """Convenience function to route subgoals."""
    return get_meta_router().route_subgoals(goal_id, current_state, goal_description, tags) 