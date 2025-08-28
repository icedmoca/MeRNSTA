#!/usr/bin/env python3
"""
MetaMonitor - Meta-cognitive monitoring and improvement tracking

Tracks high-level cognitive metrics and suggests improvement goals
for the overall system performance.

Features:
- Drift event frequency tracking
- Reflex score trend analysis
- Anticipatory vs reactive repair ratio
- System-wide improvement suggestions
- Meta-cognitive performance monitoring
"""

import time
import logging
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from storage.reflex_log import get_reflex_logger
from storage.enhanced_memory_system import EnhancedMemorySystem
from agents.daemon_drift_watcher import get_drift_watcher_status
from agents.strategy_optimizer import get_performance_summary
from agents.memory_consolidator import get_consolidation_status
from config.settings import get_config


@dataclass
class MetaMetric:
    """Represents a meta-cognitive metric."""
    metric_name: str
    value: float
    timestamp: float
    trend: str  # increasing, decreasing, stable
    description: str


@dataclass
class ImprovementGoal:
    """Represents a system improvement goal."""
    goal_id: str
    goal_type: str  # drift_prevention, strategy_optimization, memory_consolidation
    description: str
    priority: float
    created_time: float
    status: str = "pending"  # pending, in_progress, completed, failed


class MetaMonitor:
    """
    Monitors meta-cognitive performance and suggests improvements.
    
    Features:
    - Drift event frequency tracking
    - Reflex score trend analysis
    - Anticipatory vs reactive repair ratio
    - System-wide improvement suggestions
    - Meta-cognitive performance monitoring
    """
    
    def __init__(self, memory_system: EnhancedMemorySystem = None):
        self.memory_system = memory_system or EnhancedMemorySystem()
        self.reflex_logger = get_reflex_logger()
        
        # Configuration
        config = get_config()
        self.monitoring_interval = config.get('meta_monitor_interval', 1800)  # 30 minutes
        self.metric_history_window = config.get('metric_history_window', 86400)  # 24 hours
        
        # Monitoring state
        self.running = False
        self.monitoring_thread = None
        self.last_monitoring_time = time.time()
        self.metrics_history: List[MetaMetric] = []
        self.improvement_goals: Dict[str, ImprovementGoal] = {}
        
        # Performance tracking
        self.monitoring_count = 0
        self.goals_generated = 0
        
        print(f"[MetaMonitor] Initialized with interval={self.monitoring_interval}s")
    
    def start_monitoring(self):
        """Start the meta-cognitive monitoring daemon."""
        if self.running:
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print(f"[MetaMonitor] Started meta-cognitive monitoring daemon")
    
    def stop_monitoring(self):
        """Stop the meta-cognitive monitoring daemon."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print(f"[MetaMonitor] Stopped meta-cognitive monitoring daemon")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                self._collect_meta_metrics()
                self._analyze_trends()
                self._generate_improvement_goals()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logging.error(f"[MetaMonitor] Error in monitoring loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _collect_meta_metrics(self):
        """Collect meta-cognitive metrics."""
        try:
            print(f"[MetaMonitor] Collecting meta-cognitive metrics...")
            
            current_time = time.time()
            metrics = []
            
            # 1. Drift event frequency (per hour)
            drift_frequency = self._calculate_drift_frequency()
            metrics.append(MetaMetric(
                metric_name="drift_events_per_hour",
                value=drift_frequency,
                timestamp=current_time,
                trend=self._determine_trend("drift_events_per_hour", drift_frequency),
                description=f"Drift events detected per hour: {drift_frequency:.2f}"
            ))
            
            # 2. Average reflex score trend
            avg_reflex_score = self._calculate_average_reflex_score()
            metrics.append(MetaMetric(
                metric_name="average_reflex_score",
                value=avg_reflex_score,
                timestamp=current_time,
                trend=self._determine_trend("average_reflex_score", avg_reflex_score),
                description=f"Average reflex score: {avg_reflex_score:.3f}"
            ))
            
            # 3. Anticipatory vs reactive repair ratio
            anticipatory_ratio = self._calculate_anticipatory_ratio()
            metrics.append(MetaMetric(
                metric_name="anticipatory_repair_ratio",
                value=anticipatory_ratio,
                timestamp=current_time,
                trend=self._determine_trend("anticipatory_repair_ratio", anticipatory_ratio),
                description=f"Anticipatory repair ratio: {anticipatory_ratio:.3f}"
            ))
            
            # 4. Strategy effectiveness
            strategy_effectiveness = self._calculate_strategy_effectiveness()
            metrics.append(MetaMetric(
                metric_name="strategy_effectiveness",
                value=strategy_effectiveness,
                timestamp=current_time,
                trend=self._determine_trend("strategy_effectiveness", strategy_effectiveness),
                description=f"Overall strategy effectiveness: {strategy_effectiveness:.3f}"
            ))
            
            # 5. Memory consolidation rate
            consolidation_rate = self._calculate_consolidation_rate()
            metrics.append(MetaMetric(
                metric_name="memory_consolidation_rate",
                value=consolidation_rate,
                timestamp=current_time,
                trend=self._determine_trend("memory_consolidation_rate", consolidation_rate),
                description=f"Memory consolidation rate: {consolidation_rate:.3f}"
            ))
            
            # Add metrics to history
            self.metrics_history.extend(metrics)
            
            # Clean old metrics
            cutoff_time = current_time - self.metric_history_window
            self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_time]
            
            self.last_monitoring_time = current_time
            self.monitoring_count += 1
            
            print(f"[MetaMonitor] Collected {len(metrics)} metrics")
            
        except Exception as e:
            logging.error(f"[MetaMonitor] Error collecting metrics: {e}")
    
    def _calculate_drift_frequency(self) -> float:
        """Calculate drift events per hour."""
        try:
            # Get recent reflex cycles (last hour)
            recent_cycles = self.reflex_logger.get_recent_cycles(limit=100)
            one_hour_ago = time.time() - 3600
            
            recent_drifts = [c for c in recent_cycles if c.start_time > one_hour_ago]
            
            return len(recent_drifts)
            
        except Exception as e:
            logging.error(f"[MetaMonitor] Error calculating drift frequency: {e}")
            return 0.0
    
    def _calculate_average_reflex_score(self) -> float:
        """Calculate average reflex score."""
        try:
            recent_scores = self.reflex_logger.get_reflex_scores(limit=50)
            
            if not recent_scores:
                return 0.0
            
            avg_score = sum(s.score for s in recent_scores) / len(recent_scores)
            return avg_score
            
        except Exception as e:
            logging.error(f"[MetaMonitor] Error calculating average reflex score: {e}")
            return 0.0
    
    def _calculate_anticipatory_ratio(self) -> float:
        """Calculate ratio of anticipatory vs reactive repairs."""
        try:
            recent_cycles = self.reflex_logger.get_recent_cycles(limit=100)
            
            if not recent_cycles:
                return 0.0
            
            anticipatory_count = sum(1 for c in recent_cycles if c.strategy == "anticipatory_drift")
            total_count = len(recent_cycles)
            
            return anticipatory_count / total_count if total_count > 0 else 0.0
            
        except Exception as e:
            logging.error(f"[MetaMonitor] Error calculating anticipatory ratio: {e}")
            return 0.0
    
    def _calculate_strategy_effectiveness(self) -> float:
        """Calculate overall strategy effectiveness."""
        try:
            performance_summary = get_performance_summary()
            
            if not performance_summary:
                return 0.0
            
            return performance_summary.get('overall_average_score', 0.0)
            
        except Exception as e:
            logging.error(f"[MetaMonitor] Error calculating strategy effectiveness: {e}")
            return 0.0
    
    def _calculate_consolidation_rate(self) -> float:
        """Calculate memory consolidation rate."""
        try:
            consolidation_status = get_consolidation_status()
            
            if not consolidation_status:
                return 0.0
            
            belief_facts_count = consolidation_status.get('belief_facts_created', 0)
            consolidation_count = consolidation_status.get('consolidation_count', 0)
            
            if consolidation_count == 0:
                return 0.0
            
            return belief_facts_count / consolidation_count
            
        except Exception as e:
            logging.error(f"[MetaMonitor] Error calculating consolidation rate: {e}")
            return 0.0
    
    def _determine_trend(self, metric_name: str, current_value: float) -> str:
        """Determine trend for a metric."""
        try:
            # Get historical values for this metric
            historical_values = [
                m.value for m in self.metrics_history 
                if m.metric_name == metric_name
            ][-5:]  # Last 5 values
            
            if len(historical_values) < 2:
                return "stable"
            
            # Calculate trend
            recent_avg = sum(historical_values) / len(historical_values)
            
            if current_value > recent_avg * 1.1:
                return "increasing"
            elif current_value < recent_avg * 0.9:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logging.error(f"[MetaMonitor] Error determining trend: {e}")
            return "stable"
    
    def _analyze_trends(self):
        """Analyze metric trends and identify patterns."""
        try:
            # Analyze recent metrics for patterns
            recent_metrics = [m for m in self.metrics_history if m.timestamp > time.time() - 3600]
            
            # Check for concerning trends
            concerning_trends = []
            
            for metric in recent_metrics:
                if metric.trend == "decreasing" and metric.value < 0.5:
                    concerning_trends.append(f"{metric.metric_name} is declining ({metric.value:.3f})")
                elif metric.trend == "increasing" and metric.metric_name == "drift_events_per_hour" and metric.value > 5:
                    concerning_trends.append(f"High drift frequency detected ({metric.value:.2f}/hour)")
            
            if concerning_trends:
                print(f"[MetaMonitor] Concerning trends detected: {', '.join(concerning_trends)}")
            
        except Exception as e:
            logging.error(f"[MetaMonitor] Error analyzing trends: {e}")
    
    def _generate_improvement_goals(self):
        """Generate improvement goals based on metrics."""
        try:
            current_time = time.time()
            goals = []
            
            # Get current metrics
            current_metrics = {m.metric_name: m.value for m in self.metrics_history[-5:]}
            
            # Generate goals based on metrics
            if current_metrics.get("drift_events_per_hour", 0) > 3:
                goals.append(ImprovementGoal(
                    goal_id=f"reduce_drift_{int(current_time)}",
                    goal_type="drift_prevention",
                    description="Reduce drift event frequency - consider increasing anticipatory monitoring",
                    priority=0.8,
                    created_time=current_time
                ))
            
            if current_metrics.get("average_reflex_score", 0) < 0.6:
                goals.append(ImprovementGoal(
                    goal_id=f"improve_reflex_{int(current_time)}",
                    goal_type="strategy_optimization",
                    description="Improve reflex score effectiveness - analyze strategy performance",
                    priority=0.9,
                    created_time=current_time
                ))
            
            if current_metrics.get("anticipatory_repair_ratio", 0) < 0.3:
                goals.append(ImprovementGoal(
                    goal_id=f"increase_anticipation_{int(current_time)}",
                    goal_type="drift_prevention",
                    description="Increase anticipatory repair ratio - enhance prediction accuracy",
                    priority=0.7,
                    created_time=current_time
                ))
            
            if current_metrics.get("memory_consolidation_rate", 0) < 0.5:
                goals.append(ImprovementGoal(
                    goal_id=f"improve_consolidation_{int(current_time)}",
                    goal_type="memory_consolidation",
                    description="Improve memory consolidation rate - identify stable clusters",
                    priority=0.6,
                    created_time=current_time
                ))
            
            # Add new goals
            for goal in goals:
                self.improvement_goals[goal.goal_id] = goal
                self.goals_generated += 1
            
            if goals:
                print(f"[MetaMonitor] Generated {len(goals)} improvement goals")
            
        except Exception as e:
            logging.error(f"[MetaMonitor] Error generating improvement goals: {e}")
    
    def get_meta_statistics(self) -> Dict[str, Any]:
        """Get meta-cognitive statistics."""
        try:
            # Get current metrics
            current_metrics = {}
            for metric in self.metrics_history[-5:]:
                current_metrics[metric.metric_name] = {
                    "value": metric.value,
                    "trend": metric.trend,
                    "description": metric.description
                }
            
            # Get improvement goals
            active_goals = [g for g in self.improvement_goals.values() if g.status == "pending"]
            
            return {
                "monitoring_status": {
                    "running": self.running,
                    "last_monitoring_time": self.last_monitoring_time,
                    "monitoring_count": self.monitoring_count,
                    "goals_generated": self.goals_generated
                },
                "current_metrics": current_metrics,
                "active_improvement_goals": len(active_goals),
                "total_goals": len(self.improvement_goals),
                "metrics_history_count": len(self.metrics_history)
            }
            
        except Exception as e:
            logging.error(f"[MetaMonitor] Error getting meta statistics: {e}")
            return {}
    
    def get_improvement_goals(self) -> List[Dict[str, Any]]:
        """Get list of improvement goals."""
        goals = []
        for goal_id, goal in self.improvement_goals.items():
            goals.append({
                "goal_id": goal_id,
                "goal_type": goal.goal_type,
                "description": goal.description,
                "priority": goal.priority,
                "status": goal.status,
                "created_time": goal.created_time
            })
        return goals


# Global meta monitor instance
_meta_monitor_instance = None


def get_meta_monitor(memory_system: EnhancedMemorySystem = None) -> MetaMonitor:
    """Get or create the global meta monitor instance."""
    global _meta_monitor_instance
    
    if _meta_monitor_instance is None:
        _meta_monitor_instance = MetaMonitor(memory_system)
    
    return _meta_monitor_instance


def start_meta_monitoring(memory_system: EnhancedMemorySystem = None):
    """Start the meta-cognitive monitoring daemon."""
    monitor = get_meta_monitor(memory_system)
    monitor.start_monitoring()


def stop_meta_monitoring():
    """Stop the meta-cognitive monitoring daemon."""
    global _meta_monitor_instance
    if _meta_monitor_instance:
        _meta_monitor_instance.stop_monitoring()


def get_meta_statistics() -> Dict[str, Any]:
    """Get meta-cognitive statistics."""
    global _meta_monitor_instance
    if _meta_monitor_instance:
        return _meta_monitor_instance.get_meta_statistics()
    return {"monitoring_status": {"running": False}} 