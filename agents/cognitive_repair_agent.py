#!/usr/bin/env python3
"""
CognitiveRepairAgent - Self-healing meta-goal generation from token drift

Automatically generates meta-goals when token clusters drift, triggering
cognitive repair and belief clarification processes.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from storage.token_graph import TokenPropagationGraph, TokenDriftEvent, TokenCluster
from storage.enhanced_memory_model import EnhancedTripletFact


@dataclass
class DriftTriggeredGoal:
    """Meta-goal triggered by token drift detection."""
    goal_id: str
    type: str = "meta_goal"
    reason: str = "Token drift"
    token_id: Optional[int] = None
    cluster_id: Optional[str] = None
    drift_score: float = 0.0
    goal: str = ""
    priority: float = 0.0
    timestamp: float = 0.0
    status: str = "pending"  # pending, active, completed, failed
    affected_facts: List[str] = None
    repair_strategy: str = ""
    
    def __post_init__(self):
        if self.affected_facts is None:
            self.affected_facts = []
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class CognitiveRepairAgent:
    """
    Autonomous agent that generates meta-goals from token drift events.
    
    Features:
    - Automatic drift detection and goal generation
    - Priority-based goal ranking
    - Multi-strategy repair approaches
    - Integration with memory system
    - Real-time drift monitoring
    """
    
    def __init__(self, token_graph: Optional[TokenPropagationGraph] = None):
        self.token_graph = token_graph
        self.drift_threshold = 0.4  # Minimum drift score to trigger goals
        self.entropy_threshold = 0.3  # Minimum entropy change to trigger goals
        self.cluster_shift_threshold = 0.5  # Minimum cluster shift to trigger goals
        
        # Goal tracking
        self.generated_goals: List[DriftTriggeredGoal] = []
        self.active_goals: Dict[str, DriftTriggeredGoal] = {}
        self.completed_goals: List[DriftTriggeredGoal] = []
        
        # Configuration
        self.max_goals_per_drift = 3  # Maximum goals per drift event
        self.goal_priority_weights = {
            "drift_score": 0.4,
            "affected_facts": 0.3,
            "cluster_size": 0.2,
            "recency": 0.1
        }
        
        print(f"[CognitiveRepairAgent] Initialized cognitive repair agent")
    
    def detect_drift_triggered_goals(self, limit: int = 100) -> List[DriftTriggeredGoal]:
        """
        Detect token drift events and generate repair meta-goals.
        
        Args:
            limit: Maximum number of recent drift events to analyze
            
        Returns:
            List of drift-triggered goals
        """
        if not self.token_graph:
            return []
        
        # Get recent drift events
        recent_events = self.token_graph.get_token_drift_events(limit=limit)
        
        goals = []
        processed_tokens = set()
        
        for event in recent_events:
            # Skip if already processed or below threshold
            if (event.token_id in processed_tokens or 
                event.drift_score < self.drift_threshold):
                continue
            
            # Generate goals for this drift event
            event_goals = self._generate_goals_for_drift(event)
            goals.extend(event_goals)
            
            processed_tokens.add(event.token_id)
            
            # Limit goals per drift event
            if len(event_goals) >= self.max_goals_per_drift:
                break
        
        # Sort by priority
        goals.sort(key=lambda g: g.priority, reverse=True)
        
        # Store generated goals
        for goal in goals:
            self.generated_goals.append(goal)
            self.active_goals[goal.goal_id] = goal
        
        return goals
    
    def _generate_goals_for_drift(self, event: TokenDriftEvent) -> List[DriftTriggeredGoal]:
        """
        Generate specific goals for a drift event.
        
        Args:
            event: Token drift event to analyze
            
        Returns:
            List of repair goals
        """
        goals = []
        
        # Get cluster information
        old_cluster = self.token_graph.clusters.get(event.old_semantic_cluster)
        new_cluster = self.token_graph.clusters.get(event.new_semantic_cluster)
        
        # Calculate priority based on multiple factors
        priority = self._calculate_goal_priority(event, old_cluster, new_cluster)
        
        # Goal 1: Clarify beliefs related to drifting token
        if event.drift_score > self.drift_threshold:
            goal1 = DriftTriggeredGoal(
                goal_id=f"clarify_token_{event.token_id}_{int(time.time())}",
                token_id=event.token_id,
                drift_score=event.drift_score,
                goal=f"clarify beliefs related to token {event.token_id}",
                priority=priority,
                affected_facts=event.affected_facts,
                repair_strategy="belief_clarification"
            )
            goals.append(goal1)
        
        # Goal 2: Reassess cluster due to drift
        if old_cluster and new_cluster:
            cluster_shift_score = self._calculate_cluster_shift_score(old_cluster, new_cluster)
            
            if cluster_shift_score > self.cluster_shift_threshold:
                goal2 = DriftTriggeredGoal(
                    goal_id=f"reassess_cluster_{event.old_semantic_cluster}_{int(time.time())}",
                    cluster_id=event.old_semantic_cluster,
                    drift_score=cluster_shift_score,
                    goal=f"reassess cluster {event.old_semantic_cluster} due to semantic drift",
                    priority=priority * 0.8,  # Slightly lower priority than token goals
                    affected_facts=event.affected_facts,
                    repair_strategy="cluster_reassessment"
                )
                goals.append(goal2)
        
        # Goal 3: Consolidate facts linked to drifting token
        if len(event.affected_facts) > 3:  # Only if many facts affected
            goal3 = DriftTriggeredGoal(
                goal_id=f"consolidate_facts_{event.token_id}_{int(time.time())}",
                token_id=event.token_id,
                drift_score=event.drift_score,
                goal=f"consolidate {len(event.affected_facts)} facts linked to drifting token {event.token_id}",
                priority=priority * 0.6,  # Lower priority for consolidation
                affected_facts=event.affected_facts,
                repair_strategy="fact_consolidation"
            )
            goals.append(goal3)
        
        return goals
    
    def _calculate_goal_priority(self, event: TokenDriftEvent, 
                               old_cluster: Optional[TokenCluster], 
                               new_cluster: Optional[TokenCluster]) -> float:
        """
        Calculate priority score for a drift-triggered goal.
        
        Args:
            event: Drift event
            old_cluster: Previous semantic cluster
            new_cluster: New semantic cluster
            
        Returns:
            Priority score (0.0 to 1.0)
        """
        # Drift score factor
        drift_factor = min(1.0, event.drift_score)
        
        # Affected facts factor
        affected_facts_factor = min(1.0, len(event.affected_facts) / 10.0)
        
        # Cluster size factor
        cluster_size_factor = 0.0
        if old_cluster:
            cluster_size_factor = min(1.0, len(old_cluster.token_ids) / 20.0)
        
        # Recency factor (more recent = higher priority)
        time_since_drift = time.time() - event.timestamp
        recency_factor = max(0.0, 1.0 - (time_since_drift / 86400))  # Decay over 24 hours
        
        # Combined priority
        priority = (
            drift_factor * self.goal_priority_weights["drift_score"] +
            affected_facts_factor * self.goal_priority_weights["affected_facts"] +
            cluster_size_factor * self.goal_priority_weights["cluster_size"] +
            recency_factor * self.goal_priority_weights["recency"]
        )
        
        return min(1.0, priority)
    
    def _calculate_cluster_shift_score(self, old_cluster: TokenCluster, 
                                     new_cluster: TokenCluster) -> float:
        """
        Calculate how much a cluster has shifted semantically.
        
        Args:
            old_cluster: Previous cluster state
            new_cluster: New cluster state
            
        Returns:
            Shift score (0.0 to 1.0)
        """
        # Calculate overlap between old and new clusters
        old_tokens = set(old_cluster.token_ids)
        new_tokens = set(new_cluster.token_ids)
        
        intersection = len(old_tokens.intersection(new_tokens))
        union = len(old_tokens.union(new_tokens))
        
        if union == 0:
            return 1.0  # Complete shift
        
        # Jaccard distance = 1 - Jaccard similarity
        jaccard_similarity = intersection / union
        shift_score = 1.0 - jaccard_similarity
        
        # Factor in fact overlap
        old_facts = set(old_cluster.fact_ids)
        new_facts = set(new_cluster.fact_ids)
        
        fact_intersection = len(old_facts.intersection(new_facts))
        fact_union = len(old_facts.union(new_facts))
        
        if fact_union > 0:
            fact_shift = 1.0 - (fact_intersection / fact_union)
            # Combine token and fact shift scores
            combined_shift = (shift_score + fact_shift) / 2.0
            return combined_shift
        
        return shift_score
    
    def get_active_goals(self) -> List[DriftTriggeredGoal]:
        """Get currently active repair goals."""
        return list(self.active_goals.values())
    
    def get_completed_goals(self) -> List[DriftTriggeredGoal]:
        """Get completed repair goals."""
        return self.completed_goals.copy()
    
    def mark_goal_completed(self, goal_id: str, completion_notes: str = ""):
        """
        Mark a goal as completed.
        
        Args:
            goal_id: ID of the goal to complete
            completion_notes: Optional notes about completion
        """
        if goal_id in self.active_goals:
            goal = self.active_goals.pop(goal_id)
            goal.status = "completed"
            goal.repair_strategy += f" | Completed: {completion_notes}"
            self.completed_goals.append(goal)
            print(f"[CognitiveRepairAgent] Goal {goal_id} marked as completed")
    
    def mark_goal_failed(self, goal_id: str, failure_reason: str = ""):
        """
        Mark a goal as failed.
        
        Args:
            goal_id: ID of the goal to mark as failed
            failure_reason: Reason for failure
        """
        if goal_id in self.active_goals:
            goal = self.active_goals.pop(goal_id)
            goal.status = "failed"
            goal.repair_strategy += f" | Failed: {failure_reason}"
            self.completed_goals.append(goal)
            print(f"[CognitiveRepairAgent] Goal {goal_id} marked as failed: {failure_reason}")
    
    def get_goals_by_strategy(self, strategy: str) -> List[DriftTriggeredGoal]:
        """
        Get goals by repair strategy.
        
        Args:
            strategy: Repair strategy to filter by
            
        Returns:
            List of goals with matching strategy
        """
        return [goal for goal in self.active_goals.values() if goal.repair_strategy == strategy]
    
    def generate_repair_report(self) -> str:
        """
        Generate a comprehensive repair activity report.
        
        Returns:
            Formatted report string
        """
        active_count = len(self.active_goals)
        completed_count = len(self.completed_goals)
        total_generated = len(self.generated_goals)
        
        lines = ["🔧 Cognitive Repair Activity Report"]
        lines.append("=" * 50)
        lines.append(f"Active goals: {active_count}")
        lines.append(f"Completed goals: {completed_count}")
        lines.append(f"Total generated: {total_generated}")
        
        # Active goals by priority
        if self.active_goals:
            lines.append("\n🎯 Active Repair Goals (by priority):")
            sorted_goals = sorted(self.active_goals.values(), key=lambda g: g.priority, reverse=True)
            
            for i, goal in enumerate(sorted_goals[:5], 1):  # Show top 5
                lines.append(f"{i}. {goal.goal} (priority: {goal.priority:.3f})")
                lines.append(f"   Strategy: {goal.repair_strategy}")
                lines.append(f"   Affected facts: {len(goal.affected_facts)}")
        
        # Recent completions
        if self.completed_goals:
            recent_completions = [g for g in self.completed_goals if g.status == "completed"]
            if recent_completions:
                lines.append(f"\n✅ Recent Completions: {len(recent_completions)}")
                for goal in recent_completions[-3:]:  # Last 3
                    lines.append(f"  - {goal.goal}")
        
        # Strategy breakdown
        strategies = defaultdict(int)
        for goal in self.active_goals.values():
            strategies[goal.repair_strategy] += 1
        
        if strategies:
            lines.append("\n📊 Repair Strategies:")
            for strategy, count in strategies.items():
                lines.append(f"  - {strategy}: {count} goals")
        
        return "\n".join(lines)
    
    def run_continuous_repair(self, interval: int = 300):
        """
        Run continuous repair monitoring and goal generation.
        
        Args:
            interval: Check interval in seconds
        """
        print(f"[CognitiveRepairAgent] Starting continuous repair monitoring (interval: {interval}s)")
        
        while True:
            try:
                # Detect new drift-triggered goals
                new_goals = self.detect_drift_triggered_goals(limit=50)
                
                if new_goals:
                    print(f"[CognitiveRepairAgent] Generated {len(new_goals)} new repair goals")
                    
                    # Log high-priority goals
                    high_priority = [g for g in new_goals if g.priority > 0.7]
                    if high_priority:
                        print(f"[CognitiveRepairAgent] {len(high_priority)} high-priority goals detected:")
                        for goal in high_priority:
                            print(f"  - {goal.goal} (priority: {goal.priority:.3f})")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("[CognitiveRepairAgent] Continuous repair monitoring stopped")
                break
            except Exception as e:
                print(f"[CognitiveRepairAgent] Error in continuous repair: {e}")
                time.sleep(interval)


# Global cognitive repair agent instance
_repair_agent_instance = None


def get_cognitive_repair_agent() -> CognitiveRepairAgent:
    """Get or create the global cognitive repair agent instance."""
    global _repair_agent_instance
    
    if _repair_agent_instance is None:
        from storage.token_graph import get_token_graph
        token_graph = get_token_graph()
        _repair_agent_instance = CognitiveRepairAgent(token_graph)
    
    return _repair_agent_instance


def detect_drift_triggered_goals(limit: int = 100) -> List[DriftTriggeredGoal]:
    """Convenience function to detect drift-triggered goals."""
    return get_cognitive_repair_agent().detect_drift_triggered_goals(limit)


def generate_repair_report() -> str:
    """Convenience function to generate repair report."""
    return get_cognitive_repair_agent().generate_repair_report() 