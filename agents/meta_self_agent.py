#!/usr/bin/env python3
"""
Meta-Self Agent for MeRNSTA Phase 27 pt 2

Autonomous introspection agent that monitors the overall cognitive system and:
- Continuously analyzes logs, memory, agent contracts, dissonance, and reflection outcomes
- Detects internal issues like performance decline, identity drift, overload, or failure patterns
- Scaffolds self-generated goals using internal health triggers
- Prioritizes cognitive health over execution throughput
- Triggers autonomous corrective actions using the goal queue or internal toolset
"""

import json
import logging
import time
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import math

from agents.base import BaseAgent
from config.settings import get_config
from storage.drive_system import Goal


@dataclass
class CognitiveHealthMetrics:
    """Snapshot of cognitive system health at a point in time."""
    timestamp: float
    overall_health_score: float  # 0-1 scale
    
    # Component health scores
    memory_health: float
    agent_performance_health: float
    dissonance_health: float
    contract_alignment_health: float
    reflection_quality_health: float
    
    # System metrics
    total_memory_facts: int
    memory_bloat_ratio: float
    average_agent_performance: float
    active_dissonance_regions: int
    unresolved_contradictions: int
    contract_drift_count: int
    
    # Anomaly indicators
    anomalies_detected: List[str]
    critical_issues: List[str]
    recommendations: List[str]
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class MetaGoal:
    """Self-generated goal for cognitive improvement."""
    goal_id: str
    goal_type: str  # "reflection", "memory_optimization", "agent_adjustment", "dissonance_resolution"
    description: str
    priority: float  # 0-1 scale
    urgency: float  # 0-1 scale  
    justification: str
    target_component: str  # Which part of the system this targets
    expected_outcome: str
    
    # Tracking
    created_at: float
    scheduled_at: Optional[float] = None
    executed_at: Optional[float] = None
    completed_at: Optional[float] = None
    outcome_verified: bool = False
    follow_up_needed: bool = False
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    def to_goal(self) -> Goal:
        """Convert to standard Goal object for task queue."""
        return Goal(
            goal_id=f"meta_{self.goal_id}",
            description=self.description,
            priority=self.priority,
            strategy="meta_cognitive",
            metadata={
                "meta_type": self.goal_type,
                "target_component": self.target_component,
                "justification": self.justification,
                "expected_outcome": self.expected_outcome,
                "urgency": self.urgency,
                "source": "meta_self_agent"
            }
        )


class MetaSelfAgent(BaseAgent):
    """
    Autonomous introspection agent for cognitive system health monitoring.
    
    Continuously monitors:
    - Agent performance and contract drift
    - Memory system health and bloat
    - Dissonance levels and contradiction patterns
    - Reflection quality and outcomes
    - System integration and lifecycle stability
    
    Generates meta-goals for:
    - Internal repair and optimization
    - Agent adjustment and evolution
    - Memory management and cleanup
    - Dissonance resolution
    - Performance improvement
    """
    
    def __init__(self):
        super().__init__("meta_self")
        
        # Load configuration
        self.config = get_config()
        self.meta_config = self.config.get('meta_self_agent', {})
        
        # Configurable thresholds (no hardcoding per user's memory)
        self.health_thresholds = self.meta_config.get('health_thresholds', {})
        self.memory_bloat_threshold = self.health_thresholds.get('memory_bloat_threshold', 0.3)
        self.agent_performance_threshold = self.health_thresholds.get('agent_performance_threshold', 0.6)
        self.dissonance_threshold = self.health_thresholds.get('dissonance_threshold', 0.7)
        self.contract_drift_threshold = self.health_thresholds.get('contract_drift_threshold', 0.4)
        self.critical_health_threshold = self.health_thresholds.get('critical_health_threshold', 0.4)
        
        # Timing configuration
        self.check_interval = self.meta_config.get('check_interval_minutes', 30) * 60
        self.deep_analysis_interval = self.meta_config.get('deep_analysis_interval_hours', 6) * 3600
        self.follow_up_interval = self.meta_config.get('follow_up_interval_hours', 2) * 3600
        
        # Goal generation configuration
        self.max_active_meta_goals = self.meta_config.get('max_active_meta_goals', 5)
        self.goal_priority_weights = self.meta_config.get('goal_priority_weights', {
            'critical_system_health': 0.9,
            'performance_decline': 0.8,
            'memory_optimization': 0.6,
            'dissonance_resolution': 0.7,
            'preventive_maintenance': 0.4
        })
        
        # State tracking
        self.health_history: deque = deque(maxlen=100)
        self.generated_goals: Dict[str, MetaGoal] = {}
        self.last_health_check: Optional[datetime] = None
        self.last_deep_analysis: Optional[datetime] = None
        self.anomaly_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # System component access
        self._agent_registry = None
        self._memory_system = None
        self._dissonance_tracker = None
        self._lifecycle_manager = None
        self._task_selector = None
        self._personality_evolver = None
        
        # Persistence
        self.log_path = Path("output/meta_self_log.jsonl")
        self.log_path.parent.mkdir(exist_ok=True)
        
        # Load persistent state
        self._load_persistent_state()
        
        logging.info(f"[{self.name}] Initialized MetaSelfAgent with {len(self.generated_goals)} existing goals")
    
    def get_agent_instructions(self) -> str:
        """Get instructions for the MetaSelfAgent."""
        return """You are the MetaSelfAgent, responsible for autonomous cognitive system health monitoring.

Your primary functions:
1. Monitor overall system health across all cognitive components
2. Detect performance decline, memory bloat, dissonance spikes, and contract drift
3. Generate meta-goals for internal improvement and repair
4. Prioritize cognitive health over execution throughput
5. Provide introspective analysis and recommendations

Focus on system-level patterns and meta-cognitive health rather than specific task execution."""
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """Generate a response focused on cognitive health analysis."""
        try:
            # Trigger immediate health check for analysis
            health_metrics = self.perform_health_check()
            
            # Analyze the message for meta-cognitive relevance
            if any(keyword in message.lower() for keyword in ['health', 'performance', 'status', 'analysis']):
                return self._generate_health_report(health_metrics)
            elif any(keyword in message.lower() for keyword in ['goals', 'recommendations', 'improvements']):
                return self._generate_goal_analysis()
            elif any(keyword in message.lower() for keyword in ['reflect', 'introspect', 'analyze']):
                return self._generate_introspective_analysis()
            else:
                # General meta-cognitive response
                return self._generate_meta_cognitive_response(message, health_metrics)
        
        except Exception as e:
            logging.error(f"[{self.name}] Error in respond: {e}")
            return f"I encountered an issue while analyzing the cognitive system: {str(e)}"
    
    def perform_health_check(self) -> CognitiveHealthMetrics:
        """Perform comprehensive cognitive system health check."""
        try:
            timestamp = time.time()
            
            # Analyze each system component
            memory_health = self._analyze_memory_health()
            agent_health = self._analyze_agent_performance()
            dissonance_health = self._analyze_dissonance_health()
            contract_health = self._analyze_contract_alignment()
            reflection_health = self._analyze_reflection_quality()
            
            # Calculate overall health score
            component_scores = [
                memory_health['score'],
                agent_health['score'],
                dissonance_health['score'],
                contract_health['score'],
                reflection_health['score']
            ]
            overall_health = statistics.mean(component_scores)
            
            # Collect anomalies and issues
            anomalies = []
            critical_issues = []
            recommendations = []
            
            for component_data in [memory_health, agent_health, dissonance_health, contract_health, reflection_health]:
                anomalies.extend(component_data.get('anomalies', []))
                critical_issues.extend(component_data.get('critical_issues', []))
                recommendations.extend(component_data.get('recommendations', []))
            
            # Create health metrics snapshot
            health_metrics = CognitiveHealthMetrics(
                timestamp=timestamp,
                overall_health_score=overall_health,
                memory_health=memory_health['score'],
                agent_performance_health=agent_health['score'],
                dissonance_health=dissonance_health['score'],
                contract_alignment_health=contract_health['score'],
                reflection_quality_health=reflection_health['score'],
                total_memory_facts=memory_health.get('total_facts', 0),
                memory_bloat_ratio=memory_health.get('bloat_ratio', 0.0),
                average_agent_performance=agent_health.get('average_performance', 0.0),
                active_dissonance_regions=dissonance_health.get('active_regions', 0),
                unresolved_contradictions=dissonance_health.get('unresolved_contradictions', 0),
                contract_drift_count=contract_health.get('drift_count', 0),
                anomalies_detected=anomalies,
                critical_issues=critical_issues,
                recommendations=recommendations
            )
            
            # Store in history
            self.health_history.append(health_metrics)
            self.last_health_check = datetime.now()
            
            # Log health check
            self._log_health_check(health_metrics)
            
            # Generate goals if needed
            if overall_health < self.critical_health_threshold or critical_issues:
                self._generate_improvement_goals(health_metrics)
            
            return health_metrics
            
        except Exception as e:
            logging.error(f"[{self.name}] Error in health check: {e}")
            raise
    
    def _analyze_memory_health(self) -> Dict[str, Any]:
        """Analyze memory system health."""
        try:
            health_data = {
                'score': 0.8,  # Default healthy score
                'anomalies': [],
                'critical_issues': [],
                'recommendations': []
            }
            
            # Get memory system if available
            if self.memory_system:
                # Analyze memory statistics
                try:
                    # Check for memory bloat
                    facts = self.memory_system.get_all_facts()
                    total_facts = len(facts)
                    
                    if total_facts > 0:
                        # Calculate bloat indicators
                        low_confidence_facts = len([f for f in facts if getattr(f, 'confidence', 1.0) < 0.3])
                        bloat_ratio = low_confidence_facts / total_facts
                        
                        health_data['total_facts'] = total_facts
                        health_data['bloat_ratio'] = bloat_ratio
                        
                        # Score based on bloat ratio
                        if bloat_ratio > self.memory_bloat_threshold:
                            health_data['score'] *= (1 - bloat_ratio)
                            health_data['anomalies'].append(f"High memory bloat ratio: {bloat_ratio:.2f}")
                            
                            if bloat_ratio > 0.5:
                                health_data['critical_issues'].append("Critical memory bloat detected")
                                health_data['recommendations'].append("Schedule memory cleanup and fact pruning")
                        
                        # Check for contradiction density
                        contradictory_facts = len([f for f in facts if getattr(f, 'contradiction_score', 0) > 0.5])
                        contradiction_density = contradictory_facts / total_facts
                        
                        if contradiction_density > 0.2:
                            health_data['score'] *= 0.8
                            health_data['anomalies'].append(f"High contradiction density: {contradiction_density:.2f}")
                        
                        # Check memory utilization efficiency
                        accessed_recently = len([f for f in facts if getattr(f, 'last_accessed', 0) > time.time() - 86400])
                        utilization = accessed_recently / total_facts if total_facts > 0 else 0
                        
                        if utilization < 0.1:
                            health_data['score'] *= 0.9
                            health_data['recommendations'].append("Consider memory consolidation for better utilization")
                
                except Exception as e:
                    logging.warning(f"[{self.name}] Error analyzing memory statistics: {e}")
                    health_data['anomalies'].append("Unable to analyze memory statistics")
                    health_data['score'] *= 0.7
            
            else:
                health_data['anomalies'].append("Memory system not accessible")
                health_data['score'] = 0.5
            
            return health_data
            
        except Exception as e:
            logging.error(f"[{self.name}] Error in memory health analysis: {e}")
            return {
                'score': 0.3,
                'anomalies': [f"Memory analysis failed: {str(e)}"],
                'critical_issues': ["Memory system analysis failure"],
                'recommendations': ["Investigate memory system connectivity"]
            }
    
    def _analyze_agent_performance(self) -> Dict[str, Any]:
        """Analyze multi-agent system performance."""
        try:
            health_data = {
                'score': 0.8,
                'anomalies': [],
                'critical_issues': [],
                'recommendations': []
            }
            
            # Get agent registry if available
            if not self._agent_registry:
                try:
                    from agents.registry import get_agent_registry
                    self._agent_registry = get_agent_registry()
                except ImportError:
                    pass
            
            if self._agent_registry:
                try:
                    agents = self._agent_registry.get_all_agents()
                    if agents:
                        performance_scores = []
                        failing_agents = []
                        
                        for agent_name, agent in agents.items():
                            if hasattr(agent, 'get_lifecycle_metrics'):
                                metrics = agent.get_lifecycle_metrics()
                                performance = metrics.get('current_performance', 0.5)
                                performance_scores.append(performance)
                                
                                if performance < self.agent_performance_threshold:
                                    failing_agents.append((agent_name, performance))
                        
                        if performance_scores:
                            avg_performance = statistics.mean(performance_scores)
                            health_data['average_performance'] = avg_performance
                            health_data['score'] = avg_performance
                            
                            if failing_agents:
                                health_data['anomalies'].extend([
                                    f"Underperforming agent: {name} ({score:.2f})" 
                                    for name, score in failing_agents
                                ])
                                
                                if len(failing_agents) > len(agents) * 0.3:
                                    health_data['critical_issues'].append("Multiple agents underperforming")
                                    health_data['recommendations'].append("Consider agent replication or contract updates")
                        
                        # Check for lifecycle instability
                        unstable_agents = []
                        for agent_name, agent in agents.items():
                            if hasattr(agent, '_confidence_history') and len(agent._confidence_history) > 5:
                                recent_confidence = [h.get('confidence', 0.5) for h in agent._confidence_history[-5:]]
                                if len(recent_confidence) > 1:
                                    confidence_variance = statistics.variance(recent_confidence)
                                    if confidence_variance > 0.1:
                                        unstable_agents.append((agent_name, confidence_variance))
                        
                        if unstable_agents:
                            health_data['anomalies'].extend([
                                f"Unstable agent: {name} (variance: {var:.3f})"
                                for name, var in unstable_agents
                            ])
                            health_data['score'] *= 0.9
                
                except Exception as e:
                    logging.warning(f"[{self.name}] Error analyzing agent performance: {e}")
                    health_data['anomalies'].append("Agent performance analysis error")
                    health_data['score'] *= 0.7
            
            else:
                health_data['anomalies'].append("Agent registry not accessible")
                health_data['score'] = 0.6
            
            return health_data
            
        except Exception as e:
            logging.error(f"[{self.name}] Error in agent performance analysis: {e}")
            return {
                'score': 0.4,
                'anomalies': [f"Agent analysis failed: {str(e)}"],
                'critical_issues': ["Agent system analysis failure"],
                'recommendations': ["Investigate agent registry connectivity"]
            }
    
    def _analyze_dissonance_health(self) -> Dict[str, Any]:
        """Analyze cognitive dissonance levels."""
        try:
            health_data = {
                'score': 0.9,
                'anomalies': [],
                'critical_issues': [],
                'recommendations': []
            }
            
            # Get dissonance tracker if available
            if not self._dissonance_tracker:
                try:
                    from agents.dissonance_tracker import get_dissonance_tracker
                    self._dissonance_tracker = get_dissonance_tracker()
                except ImportError:
                    pass
            
            if self._dissonance_tracker:
                try:
                    # Get current dissonance state
                    dissonance_regions = self._dissonance_tracker.dissonance_regions
                    
                    if dissonance_regions:
                        active_regions = len(dissonance_regions)
                        health_data['active_regions'] = active_regions
                        
                        # Calculate average pressure
                        pressure_scores = [region.pressure_score for region in dissonance_regions.values()]
                        avg_pressure = statistics.mean(pressure_scores) if pressure_scores else 0
                        
                        # Check for high dissonance
                        if avg_pressure > self.dissonance_threshold:
                            health_data['score'] = 1 - avg_pressure
                            health_data['anomalies'].append(f"High cognitive dissonance: {avg_pressure:.2f}")
                            
                            if avg_pressure > 0.9:
                                health_data['critical_issues'].append("Critical dissonance levels detected")
                                health_data['recommendations'].append("Immediate dissonance resolution required")
                        
                        # Check for unresolved contradictions
                        unresolved = len([r for r in dissonance_regions.values() 
                                        if r.duration > 24 * 3600])  # > 24 hours
                        health_data['unresolved_contradictions'] = unresolved
                        
                        if unresolved > 0:
                            health_data['score'] *= (1 - min(unresolved * 0.1, 0.5))
                            health_data['anomalies'].append(f"Unresolved contradictions: {unresolved}")
                    
                    else:
                        health_data['active_regions'] = 0
                        health_data['unresolved_contradictions'] = 0
                
                except Exception as e:
                    logging.warning(f"[{self.name}] Error analyzing dissonance: {e}")
                    health_data['anomalies'].append("Dissonance analysis error")
                    health_data['score'] *= 0.8
            
            else:
                health_data['anomalies'].append("Dissonance tracker not accessible")
                health_data['score'] = 0.7
                health_data['active_regions'] = 0
                health_data['unresolved_contradictions'] = 0
            
            return health_data
            
        except Exception as e:
            logging.error(f"[{self.name}] Error in dissonance analysis: {e}")
            return {
                'score': 0.5,
                'anomalies': [f"Dissonance analysis failed: {str(e)}"],
                'critical_issues': ["Dissonance system analysis failure"],
                'recommendations': ["Investigate dissonance tracking system"]
            }
    
    def _analyze_contract_alignment(self) -> Dict[str, Any]:
        """Analyze agent contract drift and alignment."""
        try:
            health_data = {
                'score': 0.8,
                'anomalies': [],
                'critical_issues': [],
                'recommendations': []
            }
            
            drift_count = 0
            
            if self._agent_registry:
                try:
                    agents = self._agent_registry.get_all_agents()
                    
                    for agent_name, agent in agents.items():
                        if hasattr(agent, 'contract') and agent.contract:
                            # Check contract age
                            contract_age_days = (datetime.now() - agent.contract.last_updated).days
                            if contract_age_days > 30:
                                drift_count += 1
                                health_data['anomalies'].append(f"Stale contract: {agent_name} ({contract_age_days} days)")
                            
                            # Check for performance vs contract alignment
                            if hasattr(agent, 'get_lifecycle_metrics'):
                                metrics = agent.get_lifecycle_metrics()
                                contract_alignment = metrics.get('contract_alignment', 0.5)
                                
                                if contract_alignment < self.contract_drift_threshold:
                                    drift_count += 1
                                    health_data['anomalies'].append(f"Contract drift: {agent_name} ({contract_alignment:.2f})")
                        
                        else:
                            health_data['anomalies'].append(f"Missing contract: {agent_name}")
                            drift_count += 1
                    
                    health_data['drift_count'] = drift_count
                    
                    if drift_count > 0:
                        total_agents = len(agents)
                        drift_ratio = drift_count / total_agents if total_agents > 0 else 0
                        health_data['score'] = 1 - min(drift_ratio, 0.8)
                        
                        if drift_ratio > 0.5:
                            health_data['critical_issues'].append("Multiple agents have contract drift")
                            health_data['recommendations'].append("Schedule contract updates and agent realignment")
                
                except Exception as e:
                    logging.warning(f"[{self.name}] Error analyzing contracts: {e}")
                    health_data['anomalies'].append("Contract analysis error")
                    health_data['score'] *= 0.7
            
            else:
                health_data['anomalies'].append("Cannot access agents for contract analysis")
                health_data['score'] = 0.6
            
            health_data['drift_count'] = drift_count
            return health_data
            
        except Exception as e:
            logging.error(f"[{self.name}] Error in contract analysis: {e}")
            return {
                'score': 0.4,
                'anomalies': [f"Contract analysis failed: {str(e)}"],
                'critical_issues': ["Contract system analysis failure"],
                'recommendations': ["Investigate contract management system"]
            }
    
    def _analyze_reflection_quality(self) -> Dict[str, Any]:
        """Analyze reflection system quality and outcomes."""
        try:
            health_data = {
                'score': 0.8,
                'anomalies': [],
                'critical_issues': [],
                'recommendations': []
            }
            
            # Check reflection orchestrator if available
            try:
                from agents.reflection_orchestrator import get_reflection_orchestrator
                reflection_orchestrator = get_reflection_orchestrator()
                
                if hasattr(reflection_orchestrator, 'get_reflection_metrics'):
                    metrics = reflection_orchestrator.get_reflection_metrics()
                    
                    reflection_frequency = metrics.get('weekly_reflection_count', 0)
                    if reflection_frequency < 5:  # Less than 5 reflections per week
                        health_data['score'] *= 0.8
                        health_data['anomalies'].append(f"Low reflection frequency: {reflection_frequency}/week")
                    
                    avg_quality = metrics.get('average_reflection_quality', 0.5)
                    if avg_quality < 0.6:
                        health_data['score'] *= avg_quality / 0.6
                        health_data['anomalies'].append(f"Low reflection quality: {avg_quality:.2f}")
                    
                    failed_reflections = metrics.get('failed_reflections', 0)
                    if failed_reflections > 2:
                        health_data['anomalies'].append(f"Multiple reflection failures: {failed_reflections}")
                        health_data['recommendations'].append("Investigate reflection system stability")
            
            except ImportError:
                health_data['anomalies'].append("Reflection orchestrator not available")
                health_data['score'] = 0.7
            
            return health_data
            
        except Exception as e:
            logging.error(f"[{self.name}] Error in reflection analysis: {e}")
            return {
                'score': 0.5,
                'anomalies': [f"Reflection analysis failed: {str(e)}"],
                'critical_issues': [],
                'recommendations': ["Investigate reflection system"]
            }
    
    def _generate_improvement_goals(self, health_metrics: CognitiveHealthMetrics) -> None:
        """Generate meta-goals for cognitive improvement based on health analysis."""
        try:
            new_goals = []
            current_time = time.time()
            
            # Limit active goals
            active_goals = len([g for g in self.generated_goals.values() 
                              if not g.completed_at and not g.executed_at])
            
            if active_goals >= self.max_active_meta_goals:
                logging.info(f"[{self.name}] Skipping goal generation - {active_goals} active goals (max: {self.max_active_meta_goals})")
                return
            
            # Generate goals based on critical issues
            for issue in health_metrics.critical_issues:
                goal = self._create_goal_for_issue(issue, health_metrics, priority=0.9)
                if goal:
                    new_goals.append(goal)
            
            # Generate goals based on anomalies
            for anomaly in health_metrics.anomalies_detected:
                if len(new_goals) < self.max_active_meta_goals - active_goals:
                    goal = self._create_goal_for_anomaly(anomaly, health_metrics, priority=0.6)
                    if goal:
                        new_goals.append(goal)
            
            # Generate preventive goals if system health is declining
            if len(self.health_history) > 3:
                recent_scores = [h.overall_health_score for h in list(self.health_history)[-3:]]
                if len(recent_scores) > 1:
                    trend = recent_scores[-1] - recent_scores[0]
                    if trend < -0.1:  # Declining trend
                        goal = MetaGoal(
                            goal_id=f"preventive_{int(current_time)}",
                            goal_type="preventive_maintenance",
                            description="Perform preventive cognitive maintenance due to declining health trend",
                            priority=0.7,
                            urgency=0.6,
                            justification=f"Health score declining: {trend:.3f} over recent checks",
                            target_component="system_wide",
                            expected_outcome="Stabilize cognitive health and prevent further decline",
                            created_at=current_time
                        )
                        new_goals.append(goal)
            
            # Store and queue new goals
            for goal in new_goals:
                self.generated_goals[goal.goal_id] = goal
                
                # Queue goal in task system if available
                self._queue_meta_goal(goal)
                
                logging.info(f"[{self.name}] Generated meta-goal: {goal.description}")
            
            # Persist updated goals
            self._save_persistent_state()
            
        except Exception as e:
            logging.error(f"[{self.name}] Error generating improvement goals: {e}")
    
    def _create_goal_for_issue(self, issue: str, health_metrics: CognitiveHealthMetrics, priority: float) -> Optional[MetaGoal]:
        """Create a meta-goal for a specific critical issue."""
        current_time = time.time()
        
        if "memory bloat" in issue.lower():
            return MetaGoal(
                goal_id=f"memory_cleanup_{int(current_time)}",
                goal_type="memory_optimization",
                description="Clean up memory bloat and optimize fact storage",
                priority=priority,
                urgency=0.8,
                justification=f"Critical issue detected: {issue}",
                target_component="memory_system",
                expected_outcome="Reduce memory bloat ratio below threshold",
                created_at=current_time
            )
        
        elif "dissonance" in issue.lower():
            return MetaGoal(
                goal_id=f"dissonance_resolution_{int(current_time)}",
                goal_type="dissonance_resolution",
                description="Resolve critical cognitive dissonance",
                priority=priority,
                urgency=0.9,
                justification=f"Critical issue detected: {issue}",
                target_component="dissonance_system",
                expected_outcome="Reduce dissonance levels below critical threshold",
                created_at=current_time
            )
        
        elif "agent" in issue.lower() and "underperform" in issue.lower():
            return MetaGoal(
                goal_id=f"agent_optimization_{int(current_time)}",
                goal_type="agent_adjustment",
                description="Optimize underperforming agents",
                priority=priority,
                urgency=0.7,
                justification=f"Critical issue detected: {issue}",
                target_component="agent_system",
                expected_outcome="Improve agent performance above threshold",
                created_at=current_time
            )
        
        elif "contract drift" in issue.lower():
            return MetaGoal(
                goal_id=f"contract_update_{int(current_time)}",
                goal_type="agent_adjustment",
                description="Update agent contracts to address drift",
                priority=priority,
                urgency=0.6,
                justification=f"Critical issue detected: {issue}",
                target_component="contract_system",
                expected_outcome="Realign agent contracts with current performance",
                created_at=current_time
            )
        
        return None
    
    def _create_goal_for_anomaly(self, anomaly: str, health_metrics: CognitiveHealthMetrics, priority: float) -> Optional[MetaGoal]:
        """Create a meta-goal for a detected anomaly."""
        current_time = time.time()
        
        if "contradiction density" in anomaly.lower():
            return MetaGoal(
                goal_id=f"contradiction_resolution_{int(current_time)}",
                goal_type="memory_optimization",
                description="Address high contradiction density in memory",
                priority=priority,
                urgency=0.5,
                justification=f"Anomaly detected: {anomaly}",
                target_component="memory_system",
                expected_outcome="Reduce contradiction density through reconciliation",
                created_at=current_time
            )
        
        elif "reflection" in anomaly.lower():
            return MetaGoal(
                goal_id=f"reflection_improvement_{int(current_time)}",
                goal_type="reflection",
                description="Improve reflection system quality and frequency",
                priority=priority,
                urgency=0.4,
                justification=f"Anomaly detected: {anomaly}",
                target_component="reflection_system",
                expected_outcome="Increase reflection quality and consistency",
                created_at=current_time
            )
        
        return None
    
    def _queue_meta_goal(self, meta_goal: MetaGoal) -> None:
        """Queue a meta-goal in the task system."""
        try:
            # Get task selector if available
            if not self._task_selector:
                try:
                    from agents.task_selector import get_task_selector
                    self._task_selector = get_task_selector()
                except ImportError:
                    pass
            
            if self._task_selector and hasattr(self._task_selector, 'add_task'):
                # Convert to task format
                goal = meta_goal.to_goal()
                
                # Add to task queue with #meta tag
                task_id = self._task_selector.add_task(
                    goal_text=f"#meta {goal.description}",
                    priority=self._priority_to_task_priority(meta_goal.priority),
                    urgency=meta_goal.urgency,
                    importance=meta_goal.priority,
                    estimated_effort=60,  # 1 hour default
                    **goal.metadata
                )
                
                meta_goal.scheduled_at = time.time()
                logging.info(f"[{self.name}] Queued meta-goal as task: {task_id}")
            
            else:
                logging.warning(f"[{self.name}] Task selector not available - meta-goal not queued")
        
        except Exception as e:
            logging.error(f"[{self.name}] Error queuing meta-goal: {e}")
    
    def _priority_to_task_priority(self, priority: float):
        """Convert numeric priority to task priority enum."""
        try:
            from agents.task_selector import TaskPriority
            if priority >= 0.8:
                return TaskPriority.CRITICAL
            elif priority >= 0.6:
                return TaskPriority.HIGH
            elif priority >= 0.4:
                return TaskPriority.MEDIUM
            else:
                return TaskPriority.LOW
        except ImportError:
            return "HIGH"  # Fallback
    
    def trigger_introspective_analysis(self) -> Dict[str, Any]:
        """Trigger immediate deep introspective analysis."""
        try:
            logging.info(f"[{self.name}] Starting introspective analysis")
            
            # Perform comprehensive health check
            health_metrics = self.perform_health_check()
            
            # Analyze patterns and trends
            analysis_results = {
                'timestamp': time.time(),
                'health_metrics': asdict(health_metrics),
                'trend_analysis': self._analyze_health_trends(),
                'pattern_detection': self._detect_anomaly_patterns(),
                'goal_effectiveness': self._analyze_goal_effectiveness(),
                'recommendations': self._generate_strategic_recommendations(),
                'system_insights': self._generate_system_insights()
            }
            
            # Log introspective analysis
            self._log_introspective_analysis(analysis_results)
            
            # Phase 28: Trigger autonomous planning cycle
            self._trigger_autonomous_planning(health_metrics)
            
            # Phase 29: Trigger personality evolution check
            self._trigger_personality_evolution(health_metrics)
            
            # Update timing
            self.last_deep_analysis = datetime.now()
            
            return analysis_results
            
        except Exception as e:
            logging.error(f"[{self.name}] Error in introspective analysis: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    def _analyze_health_trends(self) -> Dict[str, Any]:
        """Analyze health trends over time."""
        if len(self.health_history) < 3:
            return {'status': 'insufficient_data'}
        
        try:
            # Calculate trends for each health component
            recent_history = list(self.health_history)[-10:]  # Last 10 checks
            
            trends = {}
            for component in ['overall_health_score', 'memory_health', 'agent_performance_health', 
                            'dissonance_health', 'contract_alignment_health', 'reflection_quality_health']:
                values = [getattr(h, component) for h in recent_history]
                if len(values) > 1:
                    # Simple trend calculation
                    trend = values[-1] - values[0]
                    trends[component] = {
                        'current': values[-1],
                        'trend': trend,
                        'direction': 'improving' if trend > 0.05 else 'declining' if trend < -0.05 else 'stable'
                    }
            
            return trends
            
        except Exception as e:
            logging.error(f"[{self.name}] Error analyzing trends: {e}")
            return {'error': str(e)}
    
    def _detect_anomaly_patterns(self) -> Dict[str, Any]:
        """Detect patterns in anomalies and issues."""
        try:
            patterns = {
                'recurring_anomalies': defaultdict(int),
                'anomaly_clusters': [],
                'critical_issue_frequency': defaultdict(int)
            }
            
            # Analyze recent health history for patterns
            recent_history = list(self.health_history)[-20:]  # Last 20 checks
            
            for health_check in recent_history:
                for anomaly in health_check.anomalies_detected:
                    # Extract anomaly type (first few words)
                    anomaly_type = ' '.join(anomaly.split()[:3]).lower()
                    patterns['recurring_anomalies'][anomaly_type] += 1
                
                for issue in health_check.critical_issues:
                    issue_type = ' '.join(issue.split()[:3]).lower()
                    patterns['critical_issue_frequency'][issue_type] += 1
            
            # Identify frequently recurring issues
            frequent_anomalies = {k: v for k, v in patterns['recurring_anomalies'].items() if v >= 3}
            frequent_issues = {k: v for k, v in patterns['critical_issue_frequency'].items() if v >= 2}
            
            return {
                'frequent_anomalies': frequent_anomalies,
                'frequent_critical_issues': frequent_issues,
                'total_anomaly_types': len(patterns['recurring_anomalies']),
                'pattern_diversity': len(patterns['recurring_anomalies']) / len(recent_history) if recent_history else 0
            }
            
        except Exception as e:
            logging.error(f"[{self.name}] Error detecting patterns: {e}")
            return {'error': str(e)}
    
    def _analyze_goal_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of generated meta-goals."""
        try:
            goal_stats = {
                'total_generated': len(self.generated_goals),
                'completed': 0,
                'in_progress': 0,
                'pending': 0,
                'effectiveness_score': 0.0
            }
            
            completed_goals = []
            for goal in self.generated_goals.values():
                if goal.completed_at:
                    goal_stats['completed'] += 1
                    completed_goals.append(goal)
                elif goal.executed_at:
                    goal_stats['in_progress'] += 1
                else:
                    goal_stats['pending'] += 1
            
            # Calculate effectiveness based on completed goals
            if completed_goals:
                # Simple effectiveness heuristic: completed goals that resulted in improvement
                effectiveness_scores = []
                for goal in completed_goals:
                    # Check if system health improved after goal completion
                    goal_completion_time = goal.completed_at
                    health_after = [h for h in self.health_history 
                                  if h.timestamp > goal_completion_time]
                    
                    if health_after:
                        # Compare health before and after
                        health_before = [h for h in self.health_history 
                                       if h.timestamp < goal_completion_time]
                        
                        if health_before:
                            before_score = health_before[-1].overall_health_score
                            after_score = health_after[0].overall_health_score
                            improvement = after_score - before_score
                            effectiveness_scores.append(max(0, min(1, 0.5 + improvement)))
                
                if effectiveness_scores:
                    goal_stats['effectiveness_score'] = statistics.mean(effectiveness_scores)
            
            return goal_stats
            
        except Exception as e:
            logging.error(f"[{self.name}] Error analyzing goal effectiveness: {e}")
            return {'error': str(e)}
    
    def _generate_strategic_recommendations(self) -> List[str]:
        """Generate high-level strategic recommendations."""
        recommendations = []
        
        try:
            if len(self.health_history) > 0:
                latest_health = self.health_history[-1]
                
                # System-wide recommendations
                if latest_health.overall_health_score < 0.5:
                    recommendations.append("CRITICAL: Implement emergency cognitive stabilization protocol")
                
                if latest_health.memory_bloat_ratio > 0.4:
                    recommendations.append("Schedule comprehensive memory optimization and cleanup")
                
                if latest_health.average_agent_performance < 0.6:
                    recommendations.append("Consider multi-agent system restructuring or retraining")
                
                if latest_health.active_dissonance_regions > 5:
                    recommendations.append("Prioritize contradiction resolution and belief reconciliation")
                
                if latest_health.contract_drift_count > 2:
                    recommendations.append("Update agent contracts and realign role specifications")
                
                # Pattern-based recommendations
                if len(self.health_history) > 5:
                    recent_scores = [h.overall_health_score for h in list(self.health_history)[-5:]]
                    if all(s1 > s2 for s1, s2 in zip(recent_scores[:-1], recent_scores[1:])):
                        recommendations.append("Investigate ongoing cognitive decline - consider system reset")
                
                # Goal effectiveness recommendations
                active_goals = len([g for g in self.generated_goals.values() if not g.completed_at])
                if active_goals > self.max_active_meta_goals:
                    recommendations.append("Reduce meta-goal backlog - focus on completion over generation")
            
        except Exception as e:
            logging.error(f"[{self.name}] Error generating recommendations: {e}")
            recommendations.append(f"Error in recommendation generation: {str(e)}")
        
        return recommendations
    
    def _generate_system_insights(self) -> List[str]:
        """Generate insights about system behavior and patterns."""
        insights = []
        
        try:
            # Analyze system stability
            if len(self.health_history) > 10:
                scores = [h.overall_health_score for h in list(self.health_history)[-10:]]
                variance = statistics.variance(scores)
                
                if variance < 0.01:
                    insights.append("System shows high stability with consistent health scores")
                elif variance > 0.05:
                    insights.append("System shows high volatility - consider stabilization measures")
                
                # Analyze component correlations
                memory_scores = [h.memory_health for h in list(self.health_history)[-10:]]
                agent_scores = [h.agent_performance_health for h in list(self.health_history)[-10:]]
                
                if len(memory_scores) == len(agent_scores) and len(memory_scores) > 3:
                    # Simple correlation check
                    memory_trend = memory_scores[-1] - memory_scores[0]
                    agent_trend = agent_scores[-1] - agent_scores[0]
                    
                    if abs(memory_trend - agent_trend) < 0.1:
                        insights.append("Memory health and agent performance show correlated trends")
            
            # Analyze goal patterns
            goal_types = defaultdict(int)
            for goal in self.generated_goals.values():
                goal_types[goal.goal_type] += 1
            
            if goal_types:
                most_common_type = max(goal_types, key=goal_types.get)
                insights.append(f"Most frequent meta-goal type: {most_common_type} ({goal_types[most_common_type]} goals)")
            
        except Exception as e:
            logging.error(f"[{self.name}] Error generating insights: {e}")
            insights.append(f"Error in insight generation: {str(e)}")
        
        return insights
    
    def get_recent_meta_goals(self, limit: int = 10) -> List[MetaGoal]:
        """Get recent meta-goals for status reporting."""
        goals = sorted(self.generated_goals.values(), 
                      key=lambda g: g.created_at, reverse=True)
        return goals[:limit]
    
    def mark_goal_completed(self, goal_id: str, outcome: str = "") -> bool:
        """Mark a meta-goal as completed."""
        try:
            if goal_id in self.generated_goals:
                goal = self.generated_goals[goal_id]
                goal.completed_at = time.time()
                goal.outcome_verified = True
                
                # Log completion
                self._log_goal_completion(goal, outcome)
                
                # Save state
                self._save_persistent_state()
                
                return True
            return False
            
        except Exception as e:
            logging.error(f"[{self.name}] Error marking goal completed: {e}")
            return False
    
    def should_run_health_check(self) -> bool:
        """Check if it's time for a health check."""
        if not self.last_health_check:
            return True
        
        time_since_check = (datetime.now() - self.last_health_check).total_seconds()
        return time_since_check >= self.check_interval
    
    def should_run_deep_analysis(self) -> bool:
        """Check if it's time for deep introspective analysis."""
        if not self.last_deep_analysis:
            return True
        
        time_since_analysis = (datetime.now() - self.last_deep_analysis).total_seconds()
        return time_since_analysis >= self.deep_analysis_interval
    
    def _generate_health_report(self, health_metrics: CognitiveHealthMetrics) -> str:
        """Generate a human-readable health report."""
        report_lines = [
            f"🧠 **Cognitive System Health Report**",
            f"Generated: {datetime.fromtimestamp(health_metrics.timestamp).strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"📊 **Overall Health Score**: {health_metrics.overall_health_score:.2f}/1.0",
            "",
            "🔍 **Component Health**:",
            f"  • Memory System: {health_metrics.memory_health:.2f}",
            f"  • Agent Performance: {health_metrics.agent_performance_health:.2f}",
            f"  • Dissonance Management: {health_metrics.dissonance_health:.2f}",
            f"  • Contract Alignment: {health_metrics.contract_alignment_health:.2f}",
            f"  • Reflection Quality: {health_metrics.reflection_quality_health:.2f}",
            "",
            f"📈 **System Metrics**:",
            f"  • Total Memory Facts: {health_metrics.total_memory_facts:,}",
            f"  • Memory Bloat Ratio: {health_metrics.memory_bloat_ratio:.2f}",
            f"  • Average Agent Performance: {health_metrics.average_agent_performance:.2f}",
            f"  • Active Dissonance Regions: {health_metrics.active_dissonance_regions}",
            f"  • Unresolved Contradictions: {health_metrics.unresolved_contradictions}",
            f"  • Contract Drift Count: {health_metrics.contract_drift_count}",
        ]
        
        if health_metrics.anomalies_detected:
            report_lines.extend([
                "",
                "⚠️ **Anomalies Detected**:",
                *[f"  • {anomaly}" for anomaly in health_metrics.anomalies_detected]
            ])
        
        if health_metrics.critical_issues:
            report_lines.extend([
                "",
                "🚨 **Critical Issues**:",
                *[f"  • {issue}" for issue in health_metrics.critical_issues]
            ])
        
        if health_metrics.recommendations:
            report_lines.extend([
                "",
                "💡 **Recommendations**:",
                *[f"  • {rec}" for rec in health_metrics.recommendations]
            ])
        
        return "\n".join(report_lines)
    
    def _generate_goal_analysis(self) -> str:
        """Generate analysis of current meta-goals."""
        recent_goals = self.get_recent_meta_goals(10)
        
        report_lines = [
            f"🎯 **Meta-Goal Analysis**",
            f"Total Generated: {len(self.generated_goals)}",
            ""
        ]
        
        if recent_goals:
            report_lines.extend([
                "📋 **Recent Meta-Goals**:",
                ""
            ])
            
            for goal in recent_goals[:5]:
                status = "✅ Completed" if goal.completed_at else "🔄 In Progress" if goal.executed_at else "⏳ Pending"
                report_lines.extend([
                    f"**{goal.description}**",
                    f"  • Type: {goal.goal_type}",
                    f"  • Priority: {goal.priority:.2f}",
                    f"  • Status: {status}",
                    f"  • Target: {goal.target_component}",
                    f"  • Created: {datetime.fromtimestamp(goal.created_at).strftime('%Y-%m-%d %H:%M')}",
                    ""
                ])
        
        else:
            report_lines.append("No meta-goals generated yet.")
        
        return "\n".join(report_lines)
    
    def _generate_introspective_analysis(self) -> str:
        """Generate deep introspective analysis report."""
        analysis = self.trigger_introspective_analysis()
        
        report_lines = [
            f"🔍 **Deep Introspective Analysis**",
            f"Generated: {datetime.fromtimestamp(analysis['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Trend analysis
        trends = analysis.get('trend_analysis', {})
        if trends and 'error' not in trends:
            report_lines.extend([
                "📈 **Health Trends**:",
                ""
            ])
            
            for component, trend_data in trends.items():
                if isinstance(trend_data, dict):
                    direction = trend_data.get('direction', 'unknown')
                    current = trend_data.get('current', 0)
                    emoji = "📈" if direction == "improving" else "📉" if direction == "declining" else "➡️"
                    report_lines.append(f"  {emoji} {component.replace('_', ' ').title()}: {current:.2f} ({direction})")
            
            report_lines.append("")
        
        # Strategic recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            report_lines.extend([
                "💡 **Strategic Recommendations**:",
                *[f"  • {rec}" for rec in recommendations],
                ""
            ])
        
        # System insights
        insights = analysis.get('system_insights', [])
        if insights:
            report_lines.extend([
                "🧠 **System Insights**:",
                *[f"  • {insight}" for insight in insights],
                ""
            ])
        
        return "\n".join(report_lines)
    
    def _generate_meta_cognitive_response(self, message: str, health_metrics: CognitiveHealthMetrics) -> str:
        """Generate a general meta-cognitive response."""
        return f"""As the MetaSelfAgent, I'm continuously monitoring our cognitive system health.

Current system status: {health_metrics.overall_health_score:.2f}/1.0

I've detected {len(health_metrics.anomalies_detected)} anomalies and {len(health_metrics.critical_issues)} critical issues that may need attention.

I'm focused on maintaining cognitive health through:
- Monitoring agent performance and contract alignment
- Analyzing memory efficiency and contradiction patterns  
- Tracking dissonance levels and reflection quality
- Generating meta-goals for autonomous improvement

Would you like a detailed health report, goal analysis, or introspective reflection?"""
    
    def _log_health_check(self, health_metrics: CognitiveHealthMetrics) -> None:
        """Log health check to persistent storage."""
        try:
            log_entry = {
                'timestamp': health_metrics.timestamp,
                'type': 'health_check',
                'data': asdict(health_metrics)
            }
            
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logging.error(f"[{self.name}] Error logging health check: {e}")
    
    def _log_introspective_analysis(self, analysis_results: Dict[str, Any]) -> None:
        """Log introspective analysis to persistent storage."""
        try:
            log_entry = {
                'timestamp': analysis_results['timestamp'],
                'type': 'introspective_analysis',
                'data': analysis_results
            }
            
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logging.error(f"[{self.name}] Error logging introspective analysis: {e}")
    
    def _log_goal_completion(self, goal: MetaGoal, outcome: str) -> None:
        """Log goal completion to persistent storage."""
        try:
            log_entry = {
                'timestamp': time.time(),
                'type': 'goal_completion',
                'data': {
                    'goal_id': goal.goal_id,
                    'goal_type': goal.goal_type,
                    'description': goal.description,
                    'outcome': outcome,
                    'duration': goal.completed_at - goal.created_at,
                    'target_component': goal.target_component
                }
            }
            
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logging.error(f"[{self.name}] Error logging goal completion: {e}")
    
    def _load_persistent_state(self) -> None:
        """Load persistent state from log file."""
        try:
            if self.log_path.exists():
                goals_loaded = 0
                
                with open(self.log_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            
                            # Restore goals from goal generation logs
                            if entry.get('type') == 'goal_generation':
                                goal_data = entry.get('data', {})
                                if 'goal_id' in goal_data:
                                    goal = MetaGoal(**goal_data)
                                    self.generated_goals[goal.goal_id] = goal
                                    goals_loaded += 1
                
                logging.info(f"[{self.name}] Loaded {goals_loaded} meta-goals from persistent state")
                
        except Exception as e:
            logging.error(f"[{self.name}] Error loading persistent state: {e}")
    
    def _save_persistent_state(self) -> None:
        """Save current goal state to persistent storage."""
        try:
            # Save current goals state
            log_entry = {
                'timestamp': time.time(),
                'type': 'state_checkpoint',
                'data': {
                    'total_goals': len(self.generated_goals),
                    'active_goals': len([g for g in self.generated_goals.values() if not g.completed_at]),
                    'completed_goals': len([g for g in self.generated_goals.values() if g.completed_at]),
                    'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
                    'last_deep_analysis': self.last_deep_analysis.isoformat() if self.last_deep_analysis else None
                }
            }
            
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logging.error(f"[{self.name}] Error saving persistent state: {e}")
    
    def _trigger_autonomous_planning(self, health_metrics: CognitiveHealthMetrics) -> None:
        """
        Trigger autonomous planning cycle during deep introspective analysis.
        
        Phase 28: Integration with AutonomousPlanner for high-level system planning.
        """
        try:
            # Check if autonomous planning is enabled in config
            planner_config = self.config.get('autonomous_planner', {})
            if not planner_config.get('enabled', True):
                logging.debug(f"[{self.name}] Autonomous planning is disabled")
                return
            
            # Import and get the autonomous planner
            try:
                from agents.action_planner import get_autonomous_planner
                planner = get_autonomous_planner()
            except ImportError as e:
                logging.warning(f"[{self.name}] AutonomousPlanner not available: {e}")
                return
            
            logging.info(f"[{self.name}] Triggering autonomous planning cycle")
            
            # Trigger planning evaluation
            planning_results = planner.evaluate_and_update_plan()
            
            # Log the planning integration
            planning_log = {
                'timestamp': time.time(),
                'type': 'autonomous_planning_triggered',
                'data': {
                    'triggered_by': 'meta_self_agent_deep_analysis',
                    'system_health_score': health_metrics.overall_health_score,
                    'planning_results': planning_results,
                    'active_plans': len(planner.active_plans),
                    'goals_enqueued': planning_results.get('goals_enqueued', 0)
                }
            }
            
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(planning_log) + '\n')
            
            # Add planning insights to meta-goals if significant changes occurred
            if planning_results.get('status') == 'completed':
                new_plans = planning_results.get('new_plans_generated', 0)
                if new_plans > 0:
                    planning_goal = MetaGoal(
                        goal_id=f"planning_update_{int(time.time())}",
                        goal_type="planning",
                        description=f"Autonomous planning generated {new_plans} new plans",
                        priority=0.6,
                        urgency=0.4,
                        justification=f"Planning cycle found {new_plans} areas for system improvement",
                        target_component="autonomous_planner",
                        expected_outcome="Enhanced system planning and goal generation",
                        created_at=time.time()
                    )
                    
                    self.generated_goals[planning_goal.goal_id] = planning_goal
                    logging.info(f"[{self.name}] Generated meta-goal for planning updates: {planning_goal.goal_id}")
            
            logging.info(f"[{self.name}] Autonomous planning integration completed successfully")
            
        except Exception as e:
            logging.error(f"[{self.name}] Error triggering autonomous planning: {e}")
            
            # Log the error
            error_log = {
                'timestamp': time.time(),
                'type': 'autonomous_planning_error',
                'data': {
                    'error': str(e),
                    'triggered_by': 'meta_self_agent_deep_analysis'
                }
            }
            
            try:
                with open(self.log_path, 'a') as f:
                    f.write(json.dumps(error_log) + '\n')
            except:
                pass  # Don't fail on logging errors
    
    @property
    def personality_evolver(self):
        """Lazy-load personality evolver for memory-driven personality shifts."""
        if self._personality_evolver is None:
            try:
                from agents.personality_evolver import get_personality_evolver
                self._personality_evolver = get_personality_evolver()
            except ImportError as e:
                logging.error(f"[{self.name}] Could not load PersonalityEvolver: {e}")
                self._personality_evolver = None
        return self._personality_evolver
    
    def _trigger_personality_evolution(self, health_metrics: CognitiveHealthMetrics) -> None:
        """
        Trigger personality evolution checks during meta-reflection.
        
        This integrates Phase 29 personality evolution into the regular meta-reflection cycle,
        checking for weekly cadence triggers and major belief conflicts.
        
        Args:
            health_metrics: Current cognitive health metrics to inform evolution decisions
        """
        try:
            if not self.personality_evolver:
                logging.debug(f"[{self.name}] PersonalityEvolver not available - skipping evolution check")
                return
            
            # Check if personality evolution is enabled in config
            evolution_config = self.config.get('personality_evolution', {})
            if not evolution_config.get('enabled', True):
                logging.debug(f"[{self.name}] Personality evolution disabled in config")
                return
            
            # Track evolution trigger attempts
            evolution_triggered = False
            trigger_reasons = []
            
            # Check weekly cadence trigger
            if self.personality_evolver.check_weekly_cadence():
                logging.info(f"[{self.name}] Triggering weekly personality evolution")
                trace = self.personality_evolver.evolve_personality(
                    trigger_type="weekly_cadence",
                    lookback_days=7
                )
                if trace:
                    evolution_triggered = True
                    trigger_reasons.append("weekly_cadence")
                    logging.info(f"[{self.name}] Weekly evolution completed: {trace.evolution_id}")
            
            # Check major conflict trigger based on system health
            dissonance_pressure = health_metrics.unresolved_contradictions / max(1, health_metrics.total_memory_facts)
            if (health_metrics.dissonance_health < 0.5 or 
                dissonance_pressure > evolution_config.get('major_conflict_threshold', 0.8) or
                self.personality_evolver.check_major_conflict_trigger()):
                
                logging.info(f"[{self.name}] Triggering major conflict personality evolution (dissonance_health={health_metrics.dissonance_health:.2f}, pressure={dissonance_pressure:.2f})")
                trace = self.personality_evolver.evolve_personality(
                    trigger_type="major_conflict",
                    lookback_days=3  # Shorter lookback for conflict-driven evolution
                )
                if trace:
                    evolution_triggered = True
                    trigger_reasons.append("major_conflict")
                    logging.info(f"[{self.name}] Major conflict evolution completed: {trace.evolution_id}")
            
            # Check for system health-driven evolution
            if health_metrics.overall_health_score < 0.6 and not evolution_triggered:
                logging.info(f"[{self.name}] Triggering health-driven personality evolution (health_score={health_metrics.overall_health_score:.2f})")
                trace = self.personality_evolver.evolve_personality(
                    trigger_type="system_health",
                    lookback_days=5
                )
                if trace:
                    evolution_triggered = True
                    trigger_reasons.append("system_health")
                    logging.info(f"[{self.name}] Health-driven evolution completed: {trace.evolution_id}")
            
            # Log evolution activity for monitoring
            if evolution_triggered:
                evolution_log = {
                    'timestamp': time.time(),
                    'type': 'personality_evolution_triggered',
                    'triggers': trigger_reasons,
                    'health_context': {
                        'overall_health': health_metrics.overall_health_score,
                        'dissonance_health': health_metrics.dissonance_health,
                        'unresolved_contradictions': health_metrics.unresolved_contradictions,
                        'memory_health': health_metrics.memory_health
                    }
                }
                
                try:
                    with open(self.log_path, 'a') as f:
                        f.write(json.dumps(evolution_log) + '\n')
                except Exception as e:
                    logging.warning(f"[{self.name}] Could not log personality evolution activity: {e}")
            
            else:
                logging.debug(f"[{self.name}] No personality evolution triggers activated")
            
        except Exception as e:
            logging.error(f"[{self.name}] Error triggering personality evolution: {e}")
            # Don't let personality evolution errors break meta-reflection
            error_log = {
                'timestamp': time.time(),
                'type': 'personality_evolution_error',
                'error': str(e),
                'health_context': {
                    'overall_health': health_metrics.overall_health_score,
                    'dissonance_health': health_metrics.dissonance_health
                }
            }
            
            try:
                with open(self.log_path, 'a') as f:
                    f.write(json.dumps(error_log) + '\n')
            except:
                pass  # Don't fail on logging errors


# Global instance accessor
_meta_self_agent = None

def get_meta_self_agent() -> MetaSelfAgent:
    """Get the global MetaSelfAgent instance."""
    global _meta_self_agent
    if _meta_self_agent is None:
        _meta_self_agent = MetaSelfAgent()
    return _meta_self_agent