#!/usr/bin/env python3
"""
Autonomous Action Planner for MeRNSTA Phase 28

Implements a high-level planning module that enables MeRNSTA to autonomously project forward,
generate multi-step action sequences, and evolve its own internal roadmap.

This planner:
- Continuously monitors internal system state (MetaSelfAgent, memory, lifecycle, contradiction pressure, goal backlog, etc.)
- Predicts likely future system needs (e.g., expected memory growth, agent drift, tool gaps)
- Generates long-term, multi-step action plans using goal scaffolding
- Scores each plan using relevance, expected impact, resource requirements, and conflict risk
- Automatically enqueues prioritized tasks/goals into the task queue, tagged with #plan
"""

import json
import logging
import time
import uuid
import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import threading
import queue

from agents.base import BaseAgent
from config.settings import get_config
from storage.drive_system import Goal

logger = logging.getLogger(__name__)


@dataclass
class PlanStep:
    """Individual step in a multi-step action plan."""
    step_id: str
    description: str
    category: str  # "memory", "agent", "system", "goal", "reflection"
    estimated_effort: int  # minutes
    dependencies: List[str]  # step_ids this depends on
    success_criteria: str
    rollback_steps: List[str]  # steps to undo this action
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class ActionPlan:
    """Long-term multi-step action plan for autonomous system improvement."""
    plan_id: str
    title: str
    description: str
    steps: List[PlanStep]
    
    # Scoring metrics
    relevance_score: float  # 0-1 how relevant to current needs
    expected_impact: float  # 0-1 expected positive impact
    resource_requirements: float  # 0-1 computational/time cost
    conflict_risk: float  # 0-1 risk of conflicting with other plans
    overall_score: float = 0.0
    
    # Execution tracking
    status: str = "pending"  # "pending", "active", "completed", "failed", "cancelled"
    current_step: Optional[str] = None
    completed_steps: List[str] = None
    failed_steps: List[str] = None
    
    # Metadata
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    last_updated: float = 0.0
    
    # Goal tracking
    generated_goals: List[str] = None  # goal_ids spawned from this plan
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.last_updated == 0.0:
            self.last_updated = time.time()
        if self.completed_steps is None:
            self.completed_steps = []
        if self.failed_steps is None:
            self.failed_steps = []
        if self.generated_goals is None:
            self.generated_goals = []
        
        # Calculate overall score if not set
        if self.overall_score == 0.0:
            self.overall_score = self._calculate_overall_score()
    
    def _calculate_overall_score(self) -> float:
        """Calculate overall plan score from component metrics."""
        # Weight the scoring components
        weights = {
            'relevance': 0.35,
            'impact': 0.30,
            'resources': 0.20,  # Lower resource requirements = higher score
            'risk': 0.15  # Lower risk = higher score
        }
        
        score = (
            weights['relevance'] * self.relevance_score +
            weights['impact'] * self.expected_impact +
            weights['resources'] * (1.0 - self.resource_requirements) +
            weights['risk'] * (1.0 - self.conflict_risk)
        )
        
        return max(0.0, min(1.0, score))


@dataclass
class SystemProjection:
    """Projection of future system state and needs."""
    projection_timeframe: str  # "short", "medium", "long"
    projected_memory_growth: float
    projected_agent_drift: float
    predicted_tool_gaps: List[str]
    anticipated_bottlenecks: List[str]
    confidence: float  # 0-1 confidence in this projection
    basis: List[str]  # what this projection is based on
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class AutonomousPlanner(BaseAgent):
    """
    High-level autonomous planning module for MeRNSTA.
    
    Continuously monitors system state and generates multi-step action plans
    for autonomous system improvement and evolution.
    """
    
    def __init__(self):
        super().__init__("autonomous_planner")
        
        # Load configuration
        self.config = get_config()
        self.planner_config = self.config.get('autonomous_planner', {})
        
        # Planning parameters (no hardcoding per user's memory)
        self.enabled = self.planner_config.get('enabled', True)
        self.planning_interval = self.planner_config.get('planning_interval_hours', 6) * 3600
        self.max_active_plans = self.planner_config.get('max_active_plans', 3)
        self.min_plan_score = self.planner_config.get('min_plan_score', 0.4)
        self.lookahead_days = self.planner_config.get('lookahead_days', 7)
        
        # Scoring weights
        self.scoring_weights = self.planner_config.get('scoring_weights', {
            'memory_health': 0.25,
            'agent_performance': 0.25,
            'system_stability': 0.20,
            'goal_achievement': 0.15,
            'innovation_potential': 0.15
        })
        
        # Future projection parameters
        self.projection_config = self.planner_config.get('projection', {})
        self.memory_growth_threshold = self.projection_config.get('memory_growth_threshold', 1.5)
        self.drift_prediction_window = self.projection_config.get('drift_prediction_window_hours', 24)
        self.confidence_threshold = self.projection_config.get('confidence_threshold', 0.6)
        
        # State tracking
        self.active_plans: Dict[str, ActionPlan] = {}
        self.completed_plans: List[ActionPlan] = []
        self.system_projections: List[SystemProjection] = []
        self.last_planning_cycle: Optional[datetime] = None
        self.planning_history: deque = deque(maxlen=50)
        
        # System component access
        self._meta_self_agent = None
        self._memory_system = None
        self._reflection_orchestrator = None
        self._task_selector = None
        self._goal_scorer = None
        
        # Persistence
        self.plan_storage_path = Path("output/action_plan.jsonl")
        self.plan_storage_path.parent.mkdir(exist_ok=True)
        
        # Background planning
        self.planning_thread = None
        self.running = False
        self.planning_queue = queue.Queue()
        
        # Load existing plans
        self._load_persistent_plans()
        
        logging.info(f"[{self.name}] Initialized with {len(self.active_plans)} active plans")
    
    def get_agent_instructions(self) -> str:
        """Get instructions for the AutonomousPlanner."""
        return """You are the AutonomousPlanner, responsible for high-level system planning and evolution.

Your primary functions:
1. Monitor internal system state across all components
2. Predict future system needs and potential issues
3. Generate multi-step action plans for system improvement
4. Score and prioritize plans based on impact and feasibility
5. Automatically enqueue high-priority goals into the task system

Focus on long-term system health, capability evolution, and autonomous improvement."""
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """Generate planning-focused responses with FastReflex integration."""
        context = context or {}
        
        try:
            # Check if FastReflexAgent should handle this first
            reflex_response = self._try_fast_reflex_first(message, context)
            if reflex_response and not reflex_response.get('deferred_to_deep'):
                return reflex_response.get('response', '')
            
            # Handle deep planning requests
            if "status" in message.lower() or "plans" in message.lower():
                return self._generate_plan_status_report()
            elif "next" in message.lower() or "step" in message.lower():
                return self._trigger_next_plan_step()
            elif "evaluate" in message.lower() or "eval" in message.lower():
                return self._trigger_plan_evaluation()
            elif "project" in message.lower() or "future" in message.lower():
                return self._generate_future_projection_report()
            else:
                # General planning response
                return self._generate_planning_analysis(message)
        
        except Exception as e:
            logging.error(f"[{self.name}] Error in respond: {e}")
            return f"Planning analysis encountered an issue: {str(e)}"
    
    def _try_fast_reflex_first(self, message: str, context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Try FastReflexAgent first for rapid response before deep planning.
        
        Returns:
            Dict with response and metadata, or None if reflex agent unavailable
        """
        try:
            # Get FastReflexAgent from registry
            from agents.registry import AgentRegistry
            registry = AgentRegistry()
            reflex_agent = registry.get_agent('fast_reflex')
            
            if not reflex_agent:
                return None
            
            # Check if reflex agent should trigger
            should_trigger, trigger_type = reflex_agent.should_trigger_reflex(message, context)
            
            if not should_trigger:
                return None
            
            # Get reflex response
            reflex_context = context.copy() if context else {}
            reflex_context['trigger_type'] = trigger_type
            reflex_context['from_autonomous_planner'] = True
            
            response = reflex_agent.respond(message, reflex_context)
            
            # Check if it was deferred back to deep planning
            if response and 'defer' in response.lower():
                return {
                    'response': response,
                    'deferred_to_deep': True,
                    'trigger_type': trigger_type
                }
            
            # Successful reflex response
            if response:
                logging.info(f"[{self.name}] FastReflex handled request: {trigger_type}")
                return {
                    'response': response,
                    'deferred_to_deep': False,
                    'trigger_type': trigger_type
                }
            
            return None
            
        except Exception as e:
            logging.error(f"[{self.name}] Error trying fast reflex: {e}")
            return None
    
    def handle_reflex_defer_signal(self, message: str, context: Dict[str, Any] = None) -> str:
        """
        Handle requests deferred from FastReflexAgent for deep planning.
        
        This method is called when FastReflexAgent determines that a request
        requires deep planning and defers it back to the AutonomousPlanner.
        """
        context = context or {}
        defer_reason = context.get('reason', 'complexity_exceeds_reflex_capability')
        
        logging.info(f"[{self.name}] Handling deferred request from FastReflex: {defer_reason}")
        
        # Enhance context with defer information
        enhanced_context = context.copy()
        enhanced_context.update({
            'deferred_from_reflex': True,
            'defer_reason': defer_reason,
            'requires_deep_analysis': True
        })
        
        # Process with full planning capabilities
        return self._generate_planning_analysis(message)
    
    def evaluate_and_update_plan(self) -> Dict[str, Any]:
        """
        Main planning cycle method - evaluates system state and updates plans.
        
        Called by MetaSelfAgent every 6 hours or on-demand.
        
        Returns:
            Planning cycle results and statistics
        """
        if not self.enabled:
            return {"status": "disabled", "message": "Autonomous planning is disabled"}
        
        try:
            cycle_start = time.time()
            logging.info(f"[{self.name}] Starting planning cycle")
            
            # Step 1: Monitor current system state
            system_state = self._monitor_system_state()
            
            # Step 2: Generate future projections
            projections = self._generate_future_projections(system_state)
            
            # Step 3: Evaluate existing plans
            self._evaluate_existing_plans(system_state)
            
            # Step 4: Generate new plans if needed
            new_plans = self._generate_new_plans(system_state, projections)
            
            # Step 5: Score and prioritize all plans
            self._score_and_prioritize_plans()
            
            # Step 6: Enqueue high-priority goals
            enqueued_goals = self._enqueue_priority_goals()
            
            # Step 7: Update plan storage
            self._save_plans_to_storage()
            
            cycle_duration = time.time() - cycle_start
            self.last_planning_cycle = datetime.now()
            
            # Create cycle summary
            results = {
                "status": "completed",
                "cycle_duration": cycle_duration,
                "active_plans": len(self.active_plans),
                "new_plans_generated": len(new_plans),
                "goals_enqueued": len(enqueued_goals),
                "system_health_score": system_state.get('overall_health', 0.0),
                "projections_generated": len(projections),
                "timestamp": cycle_start
            }
            
            # Log the cycle
            self.planning_history.append(results)
            self._log_planning_cycle(results)
            
            logging.info(f"[{self.name}] Planning cycle completed in {cycle_duration:.2f}s")
            
            return results
            
        except Exception as e:
            logging.error(f"[{self.name}] Planning cycle failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _monitor_system_state(self) -> Dict[str, Any]:
        """Monitor current internal system state across all components."""
        try:
            state = {
                'timestamp': time.time(),
                'overall_health': 0.5  # Default fallback
            }
            
            # Get MetaSelfAgent health metrics
            if not self._meta_self_agent:
                self._meta_self_agent = self._get_meta_self_agent()
            
            if self._meta_self_agent:
                health_metrics = self._meta_self_agent.perform_health_check()
                state['meta_health'] = {
                    'overall_health': health_metrics.overall_health_score,
                    'memory_health': health_metrics.memory_health,
                    'agent_performance': health_metrics.agent_performance_health,
                    'dissonance_health': health_metrics.dissonance_health,
                    'contract_alignment': health_metrics.contract_alignment_health,
                    'reflection_quality': health_metrics.reflection_quality_health,
                    'anomalies': health_metrics.anomalies_detected,
                    'critical_issues': health_metrics.critical_issues
                }
                state['overall_health'] = health_metrics.overall_health_score
            
            # Monitor memory system
            state['memory_state'] = self._monitor_memory_state()
            
            # Monitor goal backlog and task queue
            state['goal_state'] = self._monitor_goal_state()
            
            # Monitor agent lifecycle and performance
            state['agent_state'] = self._monitor_agent_state()
            
            # Monitor contradiction pressure
            state['contradiction_state'] = self._monitor_contradiction_state()
            
            return state
            
        except Exception as e:
            logging.error(f"[{self.name}] Error monitoring system state: {e}")
            return {'timestamp': time.time(), 'overall_health': 0.5, 'error': str(e)}
    
    def _generate_future_projections(self, system_state: Dict[str, Any]) -> List[SystemProjection]:
        """Generate projections of future system needs and challenges."""
        projections = []
        
        try:
            # Short-term projection (1-2 days)
            short_term = self._project_short_term_needs(system_state)
            if short_term:
                projections.append(short_term)
            
            # Medium-term projection (1 week)
            medium_term = self._project_medium_term_needs(system_state)
            if medium_term:
                projections.append(medium_term)
            
            # Long-term projection (1 month)
            long_term = self._project_long_term_needs(system_state)
            if long_term:
                projections.append(long_term)
            
            # Store projections for reference
            self.system_projections.extend(projections)
            
            # Keep only recent projections
            cutoff = time.time() - (7 * 24 * 3600)  # 7 days
            self.system_projections = [
                p for p in self.system_projections 
                if p.timestamp > cutoff
            ]
            
            return projections
            
        except Exception as e:
            logging.error(f"[{self.name}] Error generating projections: {e}")
            return []
    
    def _project_short_term_needs(self, system_state: Dict[str, Any]) -> Optional[SystemProjection]:
        """Project system needs for the next 1-2 days."""
        try:
            memory_state = system_state.get('memory_state', {})
            meta_health = system_state.get('meta_health', {})
            
            # Predict memory growth
            current_facts = memory_state.get('total_facts', 1000)
            recent_growth_rate = memory_state.get('growth_rate_per_hour', 0.0)
            projected_growth = recent_growth_rate * 48  # 2 days
            
            # Predict agent drift
            agent_performance = meta_health.get('agent_performance', 0.5)
            drift_rate = max(0.0, 0.1 - agent_performance * 0.1)  # Performance decline increases drift
            
            # Identify immediate tool gaps
            critical_issues = meta_health.get('critical_issues', [])
            tool_gaps = [issue for issue in critical_issues if 'tool' in issue.lower() or 'capability' in issue.lower()]
            
            # Identify bottlenecks
            bottlenecks = []
            if memory_state.get('bloat_ratio', 0.0) > 0.3:
                bottlenecks.append("memory_bloat")
            if agent_performance < 0.6:
                bottlenecks.append("agent_performance")
            
            # Calculate confidence
            data_points = [
                memory_state.get('has_recent_data', False),
                len(critical_issues) > 0,
                recent_growth_rate > 0
            ]
            confidence = sum(data_points) / len(data_points)
            
            return SystemProjection(
                projection_timeframe="short",
                projected_memory_growth=projected_growth / current_facts,
                projected_agent_drift=drift_rate,
                predicted_tool_gaps=tool_gaps[:3],  # Top 3
                anticipated_bottlenecks=bottlenecks,
                confidence=confidence,
                basis=["current_memory_state", "agent_health", "recent_performance"]
            )
            
        except Exception as e:
            logging.error(f"[{self.name}] Error in short-term projection: {e}")
            return None
    
    def _project_medium_term_needs(self, system_state: Dict[str, Any]) -> Optional[SystemProjection]:
        """Project system needs for the next week."""
        try:
            # Base projection on historical trends
            memory_state = system_state.get('memory_state', {})
            goal_state = system_state.get('goal_state', {})
            
            # Memory growth projection
            current_facts = memory_state.get('total_facts', 1000)
            weekly_growth = memory_state.get('growth_rate_per_hour', 0.0) * 168  # 1 week
            
            # Goal completion trajectory
            active_goals = goal_state.get('active_goals', 0)
            completion_rate = goal_state.get('completion_rate', 0.5)
            expected_goal_pressure = active_goals * (1.0 - completion_rate)
            
            # Predict tool gaps from goal patterns
            goal_types = goal_state.get('goal_types', [])
            tool_gaps = []
            if 'memory_optimization' in goal_types:
                tool_gaps.append("memory_consolidation_tools")
            if 'agent_improvement' in goal_types:
                tool_gaps.append("agent_evolution_tools")
            
            # System stress prediction
            bottlenecks = []
            if weekly_growth / current_facts > self.memory_growth_threshold:
                bottlenecks.append("memory_scaling")
            if expected_goal_pressure > 10:
                bottlenecks.append("goal_processing")
            
            confidence = 0.7  # Medium confidence for weekly projections
            
            return SystemProjection(
                projection_timeframe="medium",
                projected_memory_growth=weekly_growth / current_facts,
                projected_agent_drift=0.05,  # Baseline drift over a week
                predicted_tool_gaps=tool_gaps,
                anticipated_bottlenecks=bottlenecks,
                confidence=confidence,
                basis=["historical_trends", "goal_analysis", "growth_patterns"]
            )
            
        except Exception as e:
            logging.error(f"[{self.name}] Error in medium-term projection: {e}")
            return None
    
    def _project_long_term_needs(self, system_state: Dict[str, Any]) -> Optional[SystemProjection]:
        """Project system needs for the next month."""
        try:
            # Long-term strategic planning
            meta_health = system_state.get('meta_health', {})
            
            # Capability evolution needs
            current_health = meta_health.get('overall_health', 0.5)
            tool_gaps = [
                "advanced_reasoning_tools",
                "cross_agent_communication",
                "enhanced_memory_compression",
                "predictive_maintenance"
            ]
            
            # System evolution bottlenecks
            bottlenecks = [
                "computational_scaling",
                "knowledge_integration",
                "goal_complexity_management"
            ]
            
            # Long-term drift prediction
            base_drift = 0.15  # Expected drift over a month
            health_factor = 1.0 - current_health
            projected_drift = base_drift + (health_factor * 0.1)
            
            confidence = 0.4  # Lower confidence for long-term projections
            
            return SystemProjection(
                projection_timeframe="long",
                projected_memory_growth=2.0,  # Expect significant growth
                projected_agent_drift=projected_drift,
                predicted_tool_gaps=tool_gaps,
                anticipated_bottlenecks=bottlenecks,
                confidence=confidence,
                basis=["strategic_analysis", "capability_assessment", "evolution_patterns"]
            )
            
        except Exception as e:
            logging.error(f"[{self.name}] Error in long-term projection: {e}")
            return None
    
    def _generate_new_plans(self, system_state: Dict[str, Any], projections: List[SystemProjection]) -> List[ActionPlan]:
        """Generate new action plans based on system state and projections."""
        new_plans = []
        
        try:
            # Don't create too many plans
            if len(self.active_plans) >= self.max_active_plans:
                return new_plans
            
            # Generate plans for high-confidence projections
            for projection in projections:
                if projection.confidence >= self.confidence_threshold:
                    plan = self._create_plan_for_projection(projection, system_state)
                    if plan and plan.overall_score >= self.min_plan_score:
                        new_plans.append(plan)
            
            # Generate plans for critical issues
            meta_health = system_state.get('meta_health', {})
            critical_issues = meta_health.get('critical_issues', [])
            for issue in critical_issues[:2]:  # Top 2 critical issues
                plan = self._create_plan_for_critical_issue(issue, system_state)
                if plan and plan.overall_score >= self.min_plan_score:
                    new_plans.append(plan)
            
            # Add plans to active plans
            for plan in new_plans:
                self.active_plans[plan.plan_id] = plan
            
            return new_plans
            
        except Exception as e:
            logging.error(f"[{self.name}] Error generating new plans: {e}")
            return []
    
    def _create_plan_for_projection(self, projection: SystemProjection, system_state: Dict[str, Any]) -> Optional[ActionPlan]:
        """Create an action plan to address a specific system projection."""
        try:
            plan_id = f"projection_{projection.projection_timeframe}_{uuid.uuid4().hex[:8]}"
            
            if projection.projection_timeframe == "short":
                return self._create_short_term_plan(plan_id, projection, system_state)
            elif projection.projection_timeframe == "medium":
                return self._create_medium_term_plan(plan_id, projection, system_state)
            elif projection.projection_timeframe == "long":
                return self._create_long_term_plan(plan_id, projection, system_state)
            
            return None
            
        except Exception as e:
            logging.error(f"[{self.name}] Error creating plan for projection: {e}")
            return None
    
    def _create_short_term_plan(self, plan_id: str, projection: SystemProjection, system_state: Dict[str, Any]) -> ActionPlan:
        """Create a short-term action plan (1-2 days)."""
        steps = []
        
        # Address memory growth
        if projection.projected_memory_growth > 0.2:
            steps.append(PlanStep(
                step_id=f"{plan_id}_memory_1",
                description="Trigger memory consolidation to manage growth",
                category="memory",
                estimated_effort=30,
                dependencies=[],
                success_criteria="Memory bloat ratio reduced by 10%",
                rollback_steps=["restore_memory_backup"]
            ))
        
        # Address performance issues
        if "agent_performance" in projection.anticipated_bottlenecks:
            steps.append(PlanStep(
                step_id=f"{plan_id}_perf_1",
                description="Run agent performance diagnostics",
                category="agent",
                estimated_effort=20,
                dependencies=[],
                success_criteria="Performance issues identified and categorized",
                rollback_steps=[]
            ))
        
        # Address tool gaps
        for gap in projection.predicted_tool_gaps[:2]:
            steps.append(PlanStep(
                step_id=f"{plan_id}_tool_{gap}",
                description=f"Develop or integrate {gap}",
                category="system",
                estimated_effort=60,
                dependencies=[],
                success_criteria=f"{gap} capability added and tested",
                rollback_steps=[f"remove_{gap}_integration"]
            ))
        
        return ActionPlan(
            plan_id=plan_id,
            title=f"Short-term System Optimization",
            description=f"Address immediate needs identified in short-term projection",
            steps=steps,
            relevance_score=0.9,  # High relevance for short-term
            expected_impact=0.7,
            resource_requirements=0.4,
            conflict_risk=0.2
        )
    
    def _create_medium_term_plan(self, plan_id: str, projection: SystemProjection, system_state: Dict[str, Any]) -> ActionPlan:
        """Create a medium-term action plan (1 week)."""
        steps = []
        
        # Memory scaling preparation
        if "memory_scaling" in projection.anticipated_bottlenecks:
            steps.extend([
                PlanStep(
                    step_id=f"{plan_id}_mem_scale_1",
                    description="Analyze memory usage patterns and growth trends",
                    category="memory",
                    estimated_effort=45,
                    dependencies=[],
                    success_criteria="Memory usage analysis report generated",
                    rollback_steps=[]
                ),
                PlanStep(
                    step_id=f"{plan_id}_mem_scale_2",
                    description="Implement memory compression strategies",
                    category="memory",
                    estimated_effort=90,
                    dependencies=[f"{plan_id}_mem_scale_1"],
                    success_criteria="Memory compression reduces storage by 15%",
                    rollback_steps=["restore_memory_compression_backup"]
                )
            ])
        
        # Goal processing optimization
        if "goal_processing" in projection.anticipated_bottlenecks:
            steps.append(PlanStep(
                step_id=f"{plan_id}_goal_opt",
                description="Optimize goal queue processing and prioritization",
                category="goal",
                estimated_effort=75,
                dependencies=[],
                success_criteria="Goal processing efficiency improved by 20%",
                rollback_steps=["restore_goal_processing_config"]
            ))
        
        return ActionPlan(
            plan_id=plan_id,
            title="Medium-term System Enhancement",
            description="Prepare for and address medium-term system needs",
            steps=steps,
            relevance_score=0.8,
            expected_impact=0.8,
            resource_requirements=0.6,
            conflict_risk=0.3
        )
    
    def _create_long_term_plan(self, plan_id: str, projection: SystemProjection, system_state: Dict[str, Any]) -> ActionPlan:
        """Create a long-term action plan (1 month)."""
        steps = []
        
        # Capability evolution
        steps.extend([
            PlanStep(
                step_id=f"{plan_id}_capability_audit",
                description="Comprehensive audit of system capabilities and gaps",
                category="system",
                estimated_effort=120,
                dependencies=[],
                success_criteria="Capability audit report with gap analysis",
                rollback_steps=[]
            ),
            PlanStep(
                step_id=f"{plan_id}_capability_roadmap",
                description="Create capability development roadmap",
                category="system",
                estimated_effort=90,
                dependencies=[f"{plan_id}_capability_audit"],
                success_criteria="12-month capability roadmap created",
                rollback_steps=[]
            )
        ])
        
        # Advanced tool development
        for tool in projection.predicted_tool_gaps[:3]:
            steps.append(PlanStep(
                step_id=f"{plan_id}_advanced_{tool}",
                description=f"Research and prototype {tool}",
                category="system",
                estimated_effort=180,
                dependencies=[f"{plan_id}_capability_roadmap"],
                success_criteria=f"{tool} prototype completed and evaluated",
                rollback_steps=[f"remove_{tool}_prototype"]
            ))
        
        return ActionPlan(
            plan_id=plan_id,
            title="Long-term System Evolution",
            description="Strategic capability development and system evolution",
            steps=steps,
            relevance_score=0.6,
            expected_impact=0.9,
            resource_requirements=0.8,
            conflict_risk=0.4
        )
    
    def _create_plan_for_critical_issue(self, issue: str, system_state: Dict[str, Any]) -> Optional[ActionPlan]:
        """Create an action plan to address a critical system issue."""
        try:
            plan_id = f"critical_{uuid.uuid4().hex[:8]}"
            
            # Create targeted steps based on issue type
            steps = []
            
            if "memory" in issue.lower():
                steps.extend([
                    PlanStep(
                        step_id=f"{plan_id}_mem_diag",
                        description=f"Diagnose memory issue: {issue}",
                        category="memory",
                        estimated_effort=30,
                        dependencies=[],
                        success_criteria="Memory issue root cause identified",
                        rollback_steps=[]
                    ),
                    PlanStep(
                        step_id=f"{plan_id}_mem_fix",
                        description=f"Implement fix for memory issue",
                        category="memory",
                        estimated_effort=60,
                        dependencies=[f"{plan_id}_mem_diag"],
                        success_criteria="Memory issue resolved",
                        rollback_steps=["restore_memory_state"]
                    )
                ])
            elif "agent" in issue.lower() or "performance" in issue.lower():
                steps.extend([
                    PlanStep(
                        step_id=f"{plan_id}_agent_analyze",
                        description=f"Analyze agent issue: {issue}",
                        category="agent",
                        estimated_effort=45,
                        dependencies=[],
                        success_criteria="Agent issue analyzed and categorized",
                        rollback_steps=[]
                    ),
                    PlanStep(
                        step_id=f"{plan_id}_agent_repair",
                        description="Apply agent repair strategies",
                        category="agent",
                        estimated_effort=90,
                        dependencies=[f"{plan_id}_agent_analyze"],
                        success_criteria="Agent performance restored",
                        rollback_steps=["restore_agent_configuration"]
                    )
                ])
            else:
                # Generic system issue
                steps.append(PlanStep(
                    step_id=f"{plan_id}_generic",
                    description=f"Address critical system issue: {issue}",
                    category="system",
                    estimated_effort=120,
                    dependencies=[],
                    success_criteria="Critical issue resolved",
                    rollback_steps=["restore_system_state"]
                ))
            
            return ActionPlan(
                plan_id=plan_id,
                title=f"Critical Issue Resolution",
                description=f"Address critical system issue: {issue}",
                steps=steps,
                relevance_score=1.0,  # Maximum relevance for critical issues
                expected_impact=0.8,
                resource_requirements=0.5,
                conflict_risk=0.3
            )
            
        except Exception as e:
            logging.error(f"[{self.name}] Error creating critical issue plan: {e}")
            return None
    
    def _score_and_prioritize_plans(self) -> None:
        """Score and prioritize all active plans."""
        try:
            for plan in self.active_plans.values():
                # Recalculate scores based on current context
                plan.overall_score = plan._calculate_overall_score()
                plan.last_updated = time.time()
            
            # Sort plans by score (highest first)
            sorted_plans = sorted(
                self.active_plans.values(),
                key=lambda p: p.overall_score,
                reverse=True
            )
            
            # Update active plans dict to maintain order
            self.active_plans = {plan.plan_id: plan for plan in sorted_plans}
            
        except Exception as e:
            logging.error(f"[{self.name}] Error scoring and prioritizing plans: {e}")
    
    def _enqueue_priority_goals(self) -> List[str]:
        """Enqueue high-priority goals from active plans into the task queue."""
        enqueued_goals = []
        
        try:
            # Get task selector if available
            if not self._task_selector:
                self._task_selector = self._get_task_selector()
            
            if not self._task_selector:
                return enqueued_goals
            
            # Process top-scoring plans
            for plan in list(self.active_plans.values())[:self.max_active_plans]:
                if plan.overall_score >= self.min_plan_score and plan.status == "pending":
                    # Create goals for immediate steps (no dependencies)
                    ready_steps = [
                        step for step in plan.steps 
                        if not step.dependencies or all(
                            dep in plan.completed_steps for dep in step.dependencies
                        )
                    ]
                    
                    for step in ready_steps[:2]:  # Limit to 2 steps per plan
                        goal_id = self._enqueue_plan_step_as_goal(plan, step)
                        if goal_id:
                            enqueued_goals.append(goal_id)
                            plan.generated_goals.append(goal_id)
                    
                    # Mark plan as active if goals were enqueued
                    if enqueued_goals:
                        plan.status = "active"
                        if not plan.started_at:
                            plan.started_at = time.time()
            
            return enqueued_goals
            
        except Exception as e:
            logging.error(f"[{self.name}] Error enqueuing priority goals: {e}")
            return []
    
    def _enqueue_plan_step_as_goal(self, plan: ActionPlan, step: PlanStep) -> Optional[str]:
        """Enqueue a specific plan step as a goal in the task queue."""
        try:
            if not self._task_selector:
                return None
            
            # Create goal description with plan context
            goal_text = f"#plan {step.description}"
            
            # Calculate priority based on plan score and step importance
            priority_score = plan.overall_score * 0.8  # Convert to task priority
            
            # Add goal to task queue
            goal_id = self._task_selector.add_task(
                goal_text=goal_text,
                priority=self._score_to_priority_enum(priority_score),
                urgency=min(1.0, plan.overall_score + 0.2),
                importance=plan.overall_score,
                estimated_effort=step.estimated_effort,
                deadline=None,
                **{
                    "plan_id": plan.plan_id,
                    "step_id": step.step_id,
                    "category": step.category,
                    "source": "autonomous_planner",
                    "plan_title": plan.title
                }
            )
            
            logging.info(f"[{self.name}] Enqueued plan step as goal: {goal_id}")
            return goal_id
            
        except Exception as e:
            logging.error(f"[{self.name}] Error enqueuing plan step: {e}")
            return None
    
    def _score_to_priority_enum(self, score: float):
        """Convert numeric score to task priority enum."""
        # Import the enum locally to avoid circular dependencies
        try:
            from agents.task_selector import TaskPriority
            if score >= 0.8:
                return TaskPriority.HIGH
            elif score >= 0.6:
                return TaskPriority.MEDIUM
            elif score >= 0.4:
                return TaskPriority.LOW
            else:
                return TaskPriority.LOW
        except ImportError:
            return "MEDIUM"  # Fallback string
    
    def _evaluate_existing_plans(self, system_state: Dict[str, Any]) -> None:
        """Evaluate and update existing active plans."""
        try:
            current_time = time.time()
            plans_to_remove = []
            
            for plan_id, plan in self.active_plans.items():
                # Check if plan should be cancelled or completed
                if self._should_cancel_plan(plan, system_state):
                    plan.status = "cancelled"
                    plans_to_remove.append(plan_id)
                    continue
                
                # Update plan progress based on completed goals
                self._update_plan_progress(plan)
                
                # Check if plan is completed
                if all(step.step_id in plan.completed_steps for step in plan.steps):
                    plan.status = "completed"
                    plan.completed_at = current_time
                    plans_to_remove.append(plan_id)
                    self.completed_plans.append(plan)
            
            # Remove completed/cancelled plans from active plans
            for plan_id in plans_to_remove:
                del self.active_plans[plan_id]
            
        except Exception as e:
            logging.error(f"[{self.name}] Error evaluating existing plans: {e}")
    
    def _should_cancel_plan(self, plan: ActionPlan, system_state: Dict[str, Any]) -> bool:
        """Determine if a plan should be cancelled based on current state."""
        try:
            # Cancel if relevance has dropped significantly
            current_relevance = self._calculate_current_relevance(plan, system_state)
            if current_relevance < 0.3:
                return True
            
            # Cancel if plan has been active too long without progress
            if plan.started_at:
                active_duration = time.time() - plan.started_at
                max_duration = 7 * 24 * 3600  # 7 days
                if active_duration > max_duration and len(plan.completed_steps) == 0:
                    return True
            
            # Cancel if too many steps have failed
            failure_rate = len(plan.failed_steps) / len(plan.steps) if plan.steps else 0
            if failure_rate > 0.5:
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"[{self.name}] Error checking plan cancellation: {e}")
            return False
    
    def _calculate_current_relevance(self, plan: ActionPlan, system_state: Dict[str, Any]) -> float:
        """Calculate current relevance of a plan based on system state."""
        try:
            # Base relevance on plan category and current system needs
            relevance = plan.relevance_score * 0.5  # Start with half original relevance
            
            meta_health = system_state.get('meta_health', {})
            current_issues = meta_health.get('critical_issues', [])
            
            # Increase relevance if plan addresses current critical issues
            for step in plan.steps:
                if any(issue.lower() in step.description.lower() for issue in current_issues):
                    relevance += 0.3
                    break
            
            # Adjust based on system health
            overall_health = meta_health.get('overall_health', 0.5)
            if overall_health < 0.5:
                # Increase relevance for repair plans when health is low
                if any(cat in plan.title.lower() for cat in ['repair', 'fix', 'critical']):
                    relevance += 0.2
            
            return min(1.0, relevance)
            
        except Exception as e:
            logging.error(f"[{self.name}] Error calculating plan relevance: {e}")
            return plan.relevance_score
    
    def _update_plan_progress(self, plan: ActionPlan) -> None:
        """Update plan progress based on completed goals."""
        try:
            # Check if any generated goals have been completed
            # This is a simplified check - in a full implementation,
            # we would query the task system for goal completion status
            
            # For now, we'll simulate progress based on time
            if plan.status == "active" and plan.started_at:
                active_time = time.time() - plan.started_at
                
                # Simulate step completion over time
                expected_completion_time = sum(step.estimated_effort for step in plan.steps) * 60  # minutes to seconds
                progress_ratio = min(1.0, active_time / expected_completion_time)
                
                expected_completed_steps = int(len(plan.steps) * progress_ratio)
                
                # Mark steps as completed if they aren't already
                for i, step in enumerate(plan.steps[:expected_completed_steps]):
                    if step.step_id not in plan.completed_steps and step.step_id not in plan.failed_steps:
                        # Randomly determine if step succeeded (90% success rate)
                        if hash(step.step_id) % 10 < 9:
                            plan.completed_steps.append(step.step_id)
                        else:
                            plan.failed_steps.append(step.step_id)
            
        except Exception as e:
            logging.error(f"[{self.name}] Error updating plan progress: {e}")
    
    def _monitor_memory_state(self) -> Dict[str, Any]:
        """Monitor current memory system state."""
        try:
            # Get memory system if available
            if not self._memory_system:
                self._memory_system = self._get_memory_system()
            
            if self._memory_system and hasattr(self._memory_system, 'get_fact_count'):
                total_facts = self._memory_system.get_fact_count()
                
                # Estimate growth rate (simplified)
                growth_rate = 0.1  # Default estimate
                
                return {
                    'total_facts': total_facts,
                    'growth_rate_per_hour': growth_rate,
                    'bloat_ratio': 0.1,  # Simplified estimate
                    'has_recent_data': True
                }
            else:
                return {
                    'total_facts': 1000,  # Default estimate
                    'growth_rate_per_hour': 0.05,
                    'bloat_ratio': 0.1,
                    'has_recent_data': False
                }
                
        except Exception as e:
            logging.error(f"[{self.name}] Error monitoring memory state: {e}")
            return {'total_facts': 1000, 'growth_rate_per_hour': 0.0, 'bloat_ratio': 0.0, 'has_recent_data': False}
    
    def _monitor_goal_state(self) -> Dict[str, Any]:
        """Monitor current goal and task queue state."""
        try:
            if not self._task_selector:
                self._task_selector = self._get_task_selector()
            
            if self._task_selector and hasattr(self._task_selector, 'task_queue'):
                active_goals = len(self._task_selector.task_queue)
                completed_goals = len(getattr(self._task_selector, 'completed_tasks', []))
                
                # Estimate completion rate
                completion_rate = completed_goals / max(1, active_goals + completed_goals)
                
                # Analyze goal types (simplified)
                goal_types = ['memory_optimization', 'agent_improvement', 'system_repair']
                
                return {
                    'active_goals': active_goals,
                    'completed_goals': completed_goals,
                    'completion_rate': completion_rate,
                    'goal_types': goal_types
                }
            else:
                return {
                    'active_goals': 0,
                    'completed_goals': 0,
                    'completion_rate': 0.5,
                    'goal_types': []
                }
                
        except Exception as e:
            logging.error(f"[{self.name}] Error monitoring goal state: {e}")
            return {'active_goals': 0, 'completed_goals': 0, 'completion_rate': 0.5, 'goal_types': []}
    
    def _monitor_agent_state(self) -> Dict[str, Any]:
        """Monitor agent lifecycle and performance state."""
        try:
            # This would integrate with agent lifecycle manager
            return {
                'active_agents': 10,  # Simplified
                'performance_scores': [0.8, 0.7, 0.9],
                'drift_indicators': ['minor_drift_agent_1'],
                'lifecycle_health': 0.8
            }
            
        except Exception as e:
            logging.error(f"[{self.name}] Error monitoring agent state: {e}")
            return {'active_agents': 0, 'performance_scores': [], 'drift_indicators': [], 'lifecycle_health': 0.5}
    
    def _monitor_contradiction_state(self) -> Dict[str, Any]:
        """Monitor contradiction pressure and dissonance state."""
        try:
            # This would integrate with dissonance tracker
            return {
                'active_contradictions': 3,
                'dissonance_pressure': 0.4,
                'resolution_rate': 0.7,
                'critical_contradictions': []
            }
            
        except Exception as e:
            logging.error(f"[{self.name}] Error monitoring contradiction state: {e}")
            return {'active_contradictions': 0, 'dissonance_pressure': 0.0, 'resolution_rate': 1.0, 'critical_contradictions': []}
    
    def _get_meta_self_agent(self):
        """Get MetaSelfAgent instance."""
        try:
            from agents.meta_self_agent import get_meta_self_agent
            return get_meta_self_agent()
        except ImportError:
            return None
    
    def _get_memory_system(self):
        """Get memory system instance."""
        try:
            from storage.memory_log import MemoryLog
            return MemoryLog()
        except ImportError:
            return None
    
    def _get_task_selector(self):
        """Get task selector instance."""
        try:
            from agents.task_selector import TaskSelector
            return TaskSelector()
        except ImportError:
            return None
    
    def _generate_plan_status_report(self) -> str:
        """Generate a status report of current plans."""
        try:
            if not self.active_plans:
                return "No active plans currently. System is operating in reactive mode."
            
            report = f"📋 Autonomous Planning Status\n\n"
            report += f"Active Plans: {len(self.active_plans)}\n"
            report += f"Completed Plans: {len(self.completed_plans)}\n"
            report += f"Last Planning Cycle: {self.last_planning_cycle or 'Never'}\n\n"
            
            report += "🎯 Active Plans:\n"
            for i, plan in enumerate(list(self.active_plans.values())[:5], 1):
                progress = len(plan.completed_steps) / len(plan.steps) if plan.steps else 0
                report += f"{i}. {plan.title} (Score: {plan.overall_score:.2f}, Progress: {progress:.1%})\n"
                report += f"   Status: {plan.status}, Steps: {len(plan.steps)}\n"
            
            return report
            
        except Exception as e:
            return f"Error generating plan status: {str(e)}"
    
    def _trigger_next_plan_step(self) -> str:
        """Trigger execution of the next highest-priority plan step."""
        try:
            if not self.active_plans:
                return "No active plans to execute steps from."
            
            # Find highest-priority plan with ready steps
            for plan in self.active_plans.values():
                ready_steps = [
                    step for step in plan.steps 
                    if (step.step_id not in plan.completed_steps and 
                        step.step_id not in plan.failed_steps and
                        all(dep in plan.completed_steps for dep in step.dependencies))
                ]
                
                if ready_steps:
                    step = ready_steps[0]
                    goal_id = self._enqueue_plan_step_as_goal(plan, step)
                    if goal_id:
                        return f"✅ Enqueued next step: {step.description} (Goal ID: {goal_id})"
                    else:
                        return f"❌ Failed to enqueue step: {step.description}"
            
            return "No ready plan steps found to execute."
            
        except Exception as e:
            return f"Error triggering next plan step: {str(e)}"
    
    def _trigger_plan_evaluation(self) -> str:
        """Trigger a full plan evaluation cycle."""
        try:
            results = self.evaluate_and_update_plan()
            
            if results['status'] == 'completed':
                return f"""✅ Plan evaluation completed:
• Duration: {results['cycle_duration']:.2f}s
• Active Plans: {results['active_plans']}
• New Plans: {results['new_plans_generated']}
• Goals Enqueued: {results['goals_enqueued']}
• System Health: {results['system_health_score']:.2f}"""
            else:
                return f"❌ Plan evaluation failed: {results.get('error', 'Unknown error')}"
                
        except Exception as e:
            return f"Error triggering plan evaluation: {str(e)}"
    
    def _generate_future_projection_report(self) -> str:
        """Generate a report of future system projections."""
        try:
            if not self.system_projections:
                return "No future projections available. Run a planning cycle to generate projections."
            
            report = "🔮 Future System Projections\n\n"
            
            for projection in self.system_projections[-3:]:  # Last 3 projections
                report += f"📊 {projection.projection_timeframe.title()}-term Projection:\n"
                report += f"   Confidence: {projection.confidence:.2f}\n"
                report += f"   Memory Growth: {projection.projected_memory_growth:.1%}\n"
                report += f"   Agent Drift: {projection.projected_agent_drift:.2f}\n"
                report += f"   Tool Gaps: {', '.join(projection.predicted_tool_gaps[:3])}\n"
                report += f"   Bottlenecks: {', '.join(projection.anticipated_bottlenecks)}\n\n"
            
            return report
            
        except Exception as e:
            return f"Error generating projection report: {str(e)}"
    
    def _generate_planning_analysis(self, message: str) -> str:
        """Generate general planning analysis response."""
        try:
            # Perform a quick system state check
            system_state = self._monitor_system_state()
            overall_health = system_state.get('overall_health', 0.5)
            
            analysis = f"🤖 Autonomous Planning Analysis\n\n"
            analysis += f"System Health: {overall_health:.2f}\n"
            analysis += f"Active Plans: {len(self.active_plans)}\n"
            analysis += f"Planning Status: {'Enabled' if self.enabled else 'Disabled'}\n\n"
            
            if overall_health < 0.5:
                analysis += "⚠️ System health is below optimal. Consider triggering plan evaluation.\n"
            elif len(self.active_plans) == 0:
                analysis += "💡 No active plans. System could benefit from strategic planning.\n"
            else:
                analysis += "✅ System is actively planning and improving.\n"
            
            return analysis
            
        except Exception as e:
            return f"Error in planning analysis: {str(e)}"
    
    def _save_plans_to_storage(self) -> None:
        """Save current plans to persistent storage."""
        try:
            # Create storage entry
            storage_entry = {
                'timestamp': time.time(),
                'active_plans': [asdict(plan) for plan in self.active_plans.values()],
                'completed_plans': [asdict(plan) for plan in self.completed_plans[-10:]],  # Last 10
                'system_projections': [asdict(proj) for proj in self.system_projections[-5:]]  # Last 5
            }
            
            # Append to JSONL file
            with open(self.plan_storage_path, 'a') as f:
                f.write(json.dumps(storage_entry) + '\n')
                
        except Exception as e:
            logging.error(f"[{self.name}] Error saving plans to storage: {e}")
    
    def _load_persistent_plans(self) -> None:
        """Load plans from persistent storage."""
        try:
            if not self.plan_storage_path.exists():
                return
            
            with open(self.plan_storage_path, 'r') as f:
                lines = f.readlines()
                
                # Get the most recent entry
                if lines:
                    latest_entry = json.loads(lines[-1])
                    
                    # Restore active plans
                    for plan_data in latest_entry.get('active_plans', []):
                        plan = ActionPlan(**plan_data)
                        self.active_plans[plan.plan_id] = plan
                    
                    # Restore completed plans
                    for plan_data in latest_entry.get('completed_plans', []):
                        plan = ActionPlan(**plan_data)
                        self.completed_plans.append(plan)
                    
                    # Restore projections
                    for proj_data in latest_entry.get('system_projections', []):
                        projection = SystemProjection(**proj_data)
                        self.system_projections.append(projection)
                        
                    logging.info(f"[{self.name}] Loaded {len(self.active_plans)} active plans from storage")
                    
        except Exception as e:
            logging.error(f"[{self.name}] Error loading persistent plans: {e}")
    
    def _log_planning_cycle(self, results: Dict[str, Any]) -> None:
        """Log planning cycle results."""
        try:
            log_entry = {
                'timestamp': time.time(),
                'type': 'planning_cycle',
                'results': results
            }
            
            log_path = Path("output/autonomous_planner_log.jsonl")
            log_path.parent.mkdir(exist_ok=True)
            
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logging.error(f"[{self.name}] Error logging planning cycle: {e}")


# Global instance accessor
_autonomous_planner = None

def get_autonomous_planner() -> AutonomousPlanner:
    """Get the global AutonomousPlanner instance."""
    global _autonomous_planner
    if _autonomous_planner is None:
        _autonomous_planner = AutonomousPlanner()
    return _autonomous_planner