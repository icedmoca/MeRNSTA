#!/usr/bin/env python3
"""
🪞 Recursive Self-Inspection System for MeRNSTA Phase 2

Provides deep introspective capabilities:
- get_self_summary() function for cognitive state analysis
- Lists most volatile and reinforced beliefs
- Shows unresolved contradiction clusters
- Suggests meta-goals when cognitive drift is detected
- Supports /introspect command for interactive self-analysis
"""

import logging
import time
import json
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import statistics

from .enhanced_memory_model import EnhancedTripletFact


@dataclass
class CognitiveSnapshot:
    """Represents a snapshot of the system's cognitive state."""
    timestamp: float
    
    # Belief analysis
    total_beliefs: int
    active_beliefs: int
    volatile_beliefs: int
    reinforced_beliefs: int
    contradictory_beliefs: int
    
    # Stability metrics
    average_confidence: float
    average_volatility: float
    belief_stability_trend: str  # "improving", "declining", "stable"
    
    # Contradiction analysis
    unresolved_contradictions: int
    contradiction_clusters: int
    critical_contradictions: int
    
    # Memory efficiency
    memory_utilization: float
    access_frequency_distribution: Dict[str, int]
    oldest_belief_age_days: float
    newest_belief_age_hours: float
    
    # Perspective tracking
    tracked_perspectives: int
    perspective_conflicts: int
    trust_distribution: Dict[str, float]
    
    # Meta-cognitive indicators
    cognitive_drift_indicators: List[str]
    recommended_actions: List[str]
    system_health_score: float


@dataclass
class BeliefInsight:
    """Represents an insight about a specific belief or belief pattern."""
    insight_type: str  # "volatile", "reinforced", "contradictory", "isolated", "trending"
    description: str
    related_facts: List[EnhancedTripletFact]
    importance_score: float
    recommended_action: Optional[str] = None
    confidence: float = 0.8


class RecursiveSelfInspection:
    """
    Provides deep self-analysis capabilities for the cognitive system.
    Monitors its own thinking patterns and suggests improvements.
    """
    
    def __init__(self):
        self.inspection_history: List[CognitiveSnapshot] = []
        self.insight_cache: Dict[str, BeliefInsight] = {}
        self.drift_threshold = 0.3  # When to flag cognitive drift
        self.health_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'fair': 0.5,
            'poor': 0.3
        }
        
        print("[SelfInspection] Initialized recursive self-inspection system")
    
    def get_self_summary(self, facts: List[EnhancedTripletFact],
                        contradiction_clusters: Dict = None,
                        belief_patterns: Dict = None,
                        agent_data: Dict = None) -> CognitiveSnapshot:
        """
        Generate comprehensive self-summary of cognitive state.
        """
        print(f"[SelfInspection] Generating self-summary from {len(facts)} facts")
        
        current_time = time.time()
        
        # Basic belief analysis
        active_facts = [f for f in facts if f.active]
        volatile_facts = [f for f in facts if f.volatile or f.volatility_score > 0.5]
        reinforced_facts = [f for f in facts if f.access_count > 3 and f.confidence > 0.8]
        contradictory_facts = [f for f in facts if f.contradiction]
        
        # Calculate averages
        confidences = [f.confidence for f in active_facts if f.confidence is not None]
        volatilities = [f.volatility_score for f in active_facts if f.volatility_score is not None]
        
        avg_confidence = statistics.mean(confidences) if confidences else 0.5
        avg_volatility = statistics.mean(volatilities) if volatilities else 0.0
        
        # Analyze belief stability trend
        stability_trend = self._analyze_stability_trend(facts)
        
        # Contradiction analysis
        unresolved_contradictions = len([f for f in contradictory_facts if f.active])
        contradiction_cluster_count = len(contradiction_clusters) if contradiction_clusters else 0
        critical_contradictions = len([f for f in contradictory_facts 
                                     if f.volatility_score > 0.8 and f.active])
        
        # Memory efficiency analysis
        memory_utilization = len(active_facts) / max(1, len(facts))
        access_distribution = self._calculate_access_distribution(facts)
        
        # Age analysis
        timestamps = [f.timestamp for f in facts if f.timestamp]
        oldest_age_days = (current_time - min(timestamps)) / (24 * 3600) if timestamps else 0
        newest_age_hours = (current_time - max(timestamps)) / 3600 if timestamps else 0
        
        # Perspective analysis
        perspectives = set(f.perspective for f in facts)
        perspective_conflicts = self._count_perspective_conflicts(facts)
        trust_dist = self._calculate_trust_distribution(agent_data) if agent_data else {}
        
        # Cognitive drift detection
        drift_indicators = self._detect_cognitive_drift(facts)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            facts, volatile_facts, contradictory_facts, drift_indicators
        )
        
        # Calculate overall health score
        health_score = self._calculate_system_health_score(
            avg_confidence, avg_volatility, memory_utilization, 
            unresolved_contradictions, len(facts)
        )
        
        snapshot = CognitiveSnapshot(
            timestamp=current_time,
            total_beliefs=len(facts),
            active_beliefs=len(active_facts),
            volatile_beliefs=len(volatile_facts),
            reinforced_beliefs=len(reinforced_facts),
            contradictory_beliefs=len(contradictory_facts),
            average_confidence=avg_confidence,
            average_volatility=avg_volatility,
            belief_stability_trend=stability_trend,
            unresolved_contradictions=unresolved_contradictions,
            contradiction_clusters=contradiction_cluster_count,
            critical_contradictions=critical_contradictions,
            memory_utilization=memory_utilization,
            access_frequency_distribution=access_distribution,
            oldest_belief_age_days=oldest_age_days,
            newest_belief_age_hours=newest_age_hours,
            tracked_perspectives=len(perspectives),
            perspective_conflicts=perspective_conflicts,
            trust_distribution=trust_dist,
            cognitive_drift_indicators=drift_indicators,
            recommended_actions=recommendations,
            system_health_score=health_score
        )
        
        self.inspection_history.append(snapshot)
        
        print(f"[SelfInspection] Generated summary: {len(active_facts)} active beliefs, "
              f"health score: {health_score:.2f}, {len(drift_indicators)} drift indicators")
        
        return snapshot
    
    def _analyze_stability_trend(self, facts: List[EnhancedTripletFact]) -> str:
        """Analyze the trend in belief stability over time."""
        if len(self.inspection_history) < 3:
            return "insufficient_data"
        
        # Look at volatility trend over recent snapshots
        recent_snapshots = self.inspection_history[-3:]
        volatilities = [s.average_volatility for s in recent_snapshots]
        
        if len(volatilities) >= 2:
            if volatilities[-1] < volatilities[0] - 0.1:
                return "improving"
            elif volatilities[-1] > volatilities[0] + 0.1:
                return "declining"
            else:
                return "stable"
        
        return "stable"
    
    def _calculate_access_distribution(self, facts: List[EnhancedTripletFact]) -> Dict[str, int]:
        """Calculate distribution of access frequencies."""
        access_counts = [f.access_count for f in facts if f.access_count is not None]
        
        distribution = {
            'never_accessed': len([c for c in access_counts if c == 0]),
            'rarely_accessed': len([c for c in access_counts if 1 <= c <= 2]), 
            'moderately_accessed': len([c for c in access_counts if 3 <= c <= 5]),
            'frequently_accessed': len([c for c in access_counts if c > 5])
        }
        
        return distribution
    
    def _count_perspective_conflicts(self, facts: List[EnhancedTripletFact]) -> int:
        """Count conflicts between different perspectives."""
        conflicts = 0
        
        # Group facts by (subject, object) pair
        fact_groups = defaultdict(list)
        for fact in facts:
            key = (fact.subject.lower(), fact.object.lower())
            fact_groups[key].append(fact)
        
        # Count conflicts within each group
        for group_facts in fact_groups.values():
            perspectives = defaultdict(list)
            for fact in group_facts:
                perspectives[fact.perspective].append(fact)
            
            if len(perspectives) > 1:
                # Check for conflicting predicates across perspectives
                predicates_by_perspective = {
                    persp: set(f.predicate for f in facts)
                    for persp, facts in perspectives.items()
                }
                
                perspective_list = list(predicates_by_perspective.keys())
                for i, persp1 in enumerate(perspective_list):
                    for persp2 in perspective_list[i+1:]:
                        pred_set1 = predicates_by_perspective[persp1]
                        pred_set2 = predicates_by_perspective[persp2]
                        
                        # Check for contradictory predicates
                        for pred1 in pred_set1:
                            for pred2 in pred_set2:
                                if self._are_contradictory_predicates(pred1, pred2):
                                    conflicts += 1
        
        return conflicts
    
    def _are_contradictory_predicates(self, pred1: str, pred2: str) -> bool:
        """Check if two predicates are contradictory."""
        contradictory_pairs = [
            ('like', 'hate'), ('like', 'dislike'),
            ('love', 'hate'), ('love', 'dislike'),
            ('enjoy', 'hate'), ('enjoy', 'dislike'),
            ('want', 'reject'), ('prefer', 'dislike')
        ]
        
        pred1_clean = pred1.lower().strip()
        pred2_clean = pred2.lower().strip()
        
        for pair in contradictory_pairs:
            if (pred1_clean in pair and pred2_clean in pair and pred1_clean != pred2_clean):
                return True
        
        return False
    
    def _calculate_trust_distribution(self, agent_data: Dict) -> Dict[str, float]:
        """Calculate distribution of trust levels across agents."""
        if not agent_data or 'agents' not in agent_data:
            return {}
        
        trust_levels = [agent.get('trust_level', 0.5) for agent in agent_data['agents'].values()]
        
        if not trust_levels:
            return {}
        
        return {
            'average_trust': statistics.mean(trust_levels),
            'trust_variance': statistics.variance(trust_levels) if len(trust_levels) > 1 else 0.0,
            'high_trust_agents': len([t for t in trust_levels if t > 0.8]),
            'low_trust_agents': len([t for t in trust_levels if t < 0.3])
        }
    
    def _detect_cognitive_drift(self, facts: List[EnhancedTripletFact]) -> List[str]:
        """Detect indicators of cognitive drift or instability."""
        drift_indicators = []
        
        # High volatility indicator
        volatile_facts = [f for f in facts if f.volatility_score > 0.7]
        if len(volatile_facts) > len(facts) * 0.2:  # More than 20% volatile
            drift_indicators.append(f"High volatility: {len(volatile_facts)} facts are highly volatile")
        
        # Rapid belief changes
        recent_changes = []
        cutoff_time = time.time() - (24 * 3600)  # Last 24 hours
        for fact in facts:
            if fact.change_history:
                recent_changes.extend([
                    c for c in fact.change_history
                    if c.get('timestamp', 0) > cutoff_time
                ])
        
        if len(recent_changes) > 10:
            drift_indicators.append(f"Rapid changes: {len(recent_changes)} belief changes in 24 hours")
        
        # Contradiction buildup
        active_contradictions = [f for f in facts if f.contradiction and f.active]
        if len(active_contradictions) > 5:
            drift_indicators.append(f"Contradiction buildup: {len(active_contradictions)} unresolved contradictions")
        
        # Low confidence trend
        if len(self.inspection_history) >= 3:
            recent_confidences = [s.average_confidence for s in self.inspection_history[-3:]]
            if recent_confidences[-1] < recent_confidences[0] - 0.2:
                drift_indicators.append("Declining confidence trend detected")
        
        # Memory fragmentation
        active_facts = [f for f in facts if f.active]
        if len(active_facts) < len(facts) * 0.7:  # Less than 70% active
            drift_indicators.append(f"Memory fragmentation: only {len(active_facts)}/{len(facts)} facts are active")
        
        return drift_indicators
    
    def _generate_recommendations(self, facts: List[EnhancedTripletFact],
                                volatile_facts: List[EnhancedTripletFact],
                                contradictory_facts: List[EnhancedTripletFact],
                                drift_indicators: List[str]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Volatility recommendations
        if len(volatile_facts) > 5:
            recommendations.append(f"🔥 Address {len(volatile_facts)} volatile beliefs through clarification")
            
            # Identify most volatile topics
            volatile_topics = Counter(f.object for f in volatile_facts)
            top_topic = volatile_topics.most_common(1)[0] if volatile_topics else None
            if top_topic:
                recommendations.append(f"   → Focus on clarifying beliefs about '{top_topic[0]}' ({top_topic[1]} volatile facts)")
        
        # Contradiction recommendations
        if len(contradictory_facts) > 3:
            recommendations.append(f"⚠️ Resolve {len(contradictory_facts)} contradictory beliefs")
            
            # Find oldest contradictions
            old_contradictions = [f for f in contradictory_facts 
                                if f.timestamp and (time.time() - f.timestamp) > 7 * 24 * 3600]
            if old_contradictions:
                recommendations.append(f"   → Priority: {len(old_contradictions)} contradictions older than 1 week")
        
        # Memory optimization
        inactive_facts = [f for f in facts if not f.active]
        if len(inactive_facts) > len(facts) * 0.3:
            recommendations.append(f"🧹 Consider pruning {len(inactive_facts)} inactive beliefs")
        
        # Confidence building
        low_confidence_facts = [f for f in facts if f.confidence < 0.5 and f.active]
        if len(low_confidence_facts) > 10:
            recommendations.append(f"💪 Reinforce {len(low_confidence_facts)} low-confidence beliefs")
        
        # Drift-specific recommendations
        if drift_indicators:
            recommendations.append("🎯 Address cognitive drift indicators:")
            for indicator in drift_indicators[:3]:  # Top 3 indicators
                recommendations.append(f"   → {indicator}")
        
        # Positive reinforcement
        high_quality_facts = [f for f in facts if f.confidence > 0.8 and f.access_count > 3]
        if len(high_quality_facts) > 10:
            recommendations.append(f"✅ {len(high_quality_facts)} beliefs are well-established and stable")
        
        return recommendations
    
    def _calculate_system_health_score(self, avg_confidence: float, avg_volatility: float,
                                     memory_utilization: float, contradictions: int,
                                     total_facts: int) -> float:
        """Calculate overall system health score (0.0 - 1.0)."""
        # Weight different factors
        confidence_score = avg_confidence  # 0.0 - 1.0
        volatility_score = 1.0 - min(1.0, avg_volatility)  # Lower volatility = better
        utilization_score = memory_utilization  # Higher utilization = better
        contradiction_score = 1.0 - min(1.0, contradictions / max(1, total_facts))  # Fewer contradictions = better
        
        # Weighted combination
        health_score = (
            confidence_score * 0.3 +
            volatility_score * 0.25 +
            utilization_score * 0.2 +
            contradiction_score * 0.25
        )
        
        return min(1.0, max(0.0, health_score))
    
    def get_most_volatile_beliefs(self, facts: List[EnhancedTripletFact], limit: int = 10) -> List[BeliefInsight]:
        """Get insights about the most volatile beliefs."""
        volatile_facts = sorted([f for f in facts if f.volatile or f.volatility_score > 0.3],
                              key=lambda x: x.volatility_score, reverse=True)
        
        insights = []
        for fact in volatile_facts[:limit]:
            insight = BeliefInsight(
                insight_type="volatile",
                description=f"Highly volatile belief: '{fact.subject} {fact.predicate} {fact.object}' (volatility: {fact.volatility_score:.2f})",
                related_facts=[fact],
                importance_score=fact.volatility_score,
                recommended_action=f"Ask clarifying question about {fact.object}",
                confidence=0.9
            )
            insights.append(insight)
        
        return insights
    
    def get_most_reinforced_beliefs(self, facts: List[EnhancedTripletFact], limit: int = 10) -> List[BeliefInsight]:
        """Get insights about the most reinforced/stable beliefs."""
        reinforced_facts = sorted([f for f in facts if f.access_count > 2 and f.confidence > 0.6],
                                key=lambda x: x.confidence * (1 + f.access_count / 10), reverse=True)
        
        insights = []
        for fact in reinforced_facts[:limit]:
            reinforcement_score = fact.confidence * (1 + fact.access_count / 10)
            insight = BeliefInsight(
                insight_type="reinforced",
                description=f"Well-established belief: '{fact.subject} {fact.predicate} {fact.object}' (confidence: {fact.confidence:.2f}, accessed: {fact.access_count}x)",
                related_facts=[fact],
                importance_score=reinforcement_score,
                recommended_action="Consider consolidating this stable belief",
                confidence=fact.confidence
            )
            insights.append(insight)
        
        return insights
    
    def get_unresolved_contradictions(self, facts: List[EnhancedTripletFact]) -> List[BeliefInsight]:
        """Get insights about unresolved contradictions."""
        contradictory_facts = [f for f in facts if f.contradiction and f.active]
        
        # Group contradictions by topic
        contradiction_groups = defaultdict(list)
        for fact in contradictory_facts:
            key = (fact.subject.lower(), fact.object.lower())
            contradiction_groups[key].append(fact)
        
        insights = []
        for (subject, obj), group_facts in contradiction_groups.items():
            if len(group_facts) >= 2:
                age_days = min((time.time() - f.timestamp) / (24 * 3600) 
                             for f in group_facts if f.timestamp)
                
                insight = BeliefInsight(
                    insight_type="contradictory",
                    description=f"Unresolved contradiction about {obj}: {len(group_facts)} conflicting beliefs (age: {age_days:.1f} days)",
                    related_facts=group_facts,
                    importance_score=len(group_facts) * (1 + 1/max(1, age_days)),  # More important if recent and multiple
                    recommended_action=f"Resolve contradiction about {obj}",
                    confidence=0.8
                )
                insights.append(insight)
        
        # Sort by importance
        insights.sort(key=lambda x: x.importance_score, reverse=True)
        return insights
    
    def format_introspection_report(self, snapshot: CognitiveSnapshot,
                                  volatile_insights: List[BeliefInsight] = None,
                                  reinforced_insights: List[BeliefInsight] = None,
                                  contradiction_insights: List[BeliefInsight] = None) -> str:
        """Format a comprehensive introspection report for display."""
        
        report_lines = []
        
        # Header
        report_lines.append("🪞 RECURSIVE SELF-INSPECTION REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {time.ctime(snapshot.timestamp)}")
        report_lines.append("")
        
        # System Health Overview
        health_status = "EXCELLENT" if snapshot.system_health_score >= 0.9 else \
                       "GOOD" if snapshot.system_health_score >= 0.7 else \
                       "FAIR" if snapshot.system_health_score >= 0.5 else \
                       "NEEDS ATTENTION"
        
        report_lines.append(f"🎯 OVERALL SYSTEM HEALTH: {health_status} ({snapshot.system_health_score:.2f})")
        report_lines.append("")
        
        # Belief Statistics
        report_lines.append("📊 BELIEF STATISTICS:")
        report_lines.append(f"   Total beliefs: {snapshot.total_beliefs}")
        report_lines.append(f"   Active beliefs: {snapshot.active_beliefs}")
        report_lines.append(f"   Volatile beliefs: {snapshot.volatile_beliefs}")
        report_lines.append(f"   Reinforced beliefs: {snapshot.reinforced_beliefs}")
        report_lines.append(f"   Contradictory beliefs: {snapshot.contradictory_beliefs}")
        report_lines.append(f"   Average confidence: {snapshot.average_confidence:.2f}")
        report_lines.append(f"   Average volatility: {snapshot.average_volatility:.2f}")
        report_lines.append(f"   Stability trend: {snapshot.belief_stability_trend}")
        report_lines.append("")
        
        # Memory Efficiency
        report_lines.append("💾 MEMORY EFFICIENCY:")
        report_lines.append(f"   Memory utilization: {snapshot.memory_utilization:.1%}")
        report_lines.append(f"   Oldest belief: {snapshot.oldest_belief_age_days:.1f} days old")
        report_lines.append(f"   Newest belief: {snapshot.newest_belief_age_hours:.1f} hours old")
        report_lines.append("")
        
        # Contradiction Analysis
        if snapshot.unresolved_contradictions > 0:
            report_lines.append("⚠️ CONTRADICTION ANALYSIS:")
            report_lines.append(f"   Unresolved contradictions: {snapshot.unresolved_contradictions}")
            report_lines.append(f"   Contradiction clusters: {snapshot.contradiction_clusters}")
            report_lines.append(f"   Critical contradictions: {snapshot.critical_contradictions}")
            report_lines.append("")
        
        # Perspective Tracking
        if snapshot.tracked_perspectives > 1:
            report_lines.append("👥 PERSPECTIVE TRACKING:")
            report_lines.append(f"   Tracked perspectives: {snapshot.tracked_perspectives}")
            report_lines.append(f"   Perspective conflicts: {snapshot.perspective_conflicts}")
            if snapshot.trust_distribution:
                avg_trust = snapshot.trust_distribution.get('average_trust', 0.5)
                report_lines.append(f"   Average trust level: {avg_trust:.2f}")
            report_lines.append("")
        
        # Cognitive Drift Indicators
        if snapshot.cognitive_drift_indicators:
            report_lines.append("🌊 COGNITIVE DRIFT INDICATORS:")
            for indicator in snapshot.cognitive_drift_indicators:
                report_lines.append(f"   ⚡ {indicator}")
            report_lines.append("")
        
        # Most Volatile Beliefs
        if volatile_insights:
            report_lines.append("🔥 MOST VOLATILE BELIEFS:")
            for insight in volatile_insights[:5]:
                report_lines.append(f"   • {insight.description}")
            report_lines.append("")
        
        # Most Reinforced Beliefs
        if reinforced_insights:
            report_lines.append("💎 MOST REINFORCED BELIEFS:")
            for insight in reinforced_insights[:5]:
                report_lines.append(f"   • {insight.description}")
            report_lines.append("")
        
        # Unresolved Contradictions
        if contradiction_insights:
            report_lines.append("⚖️ UNRESOLVED CONTRADICTIONS:")
            for insight in contradiction_insights[:5]:
                report_lines.append(f"   • {insight.description}")
            report_lines.append("")
        
        # Recommendations
        if snapshot.recommended_actions:
            report_lines.append("🎯 RECOMMENDED ACTIONS:")
            for action in snapshot.recommended_actions:
                report_lines.append(f"   {action}")
            report_lines.append("")
        
        # Footer
        report_lines.append("=" * 50)
        report_lines.append("End of introspection report")
        
        return "\n".join(report_lines)
    
    def get_inspection_summary(self) -> Dict[str, Any]:
        """Get summary of inspection system state."""
        if not self.inspection_history:
            return {'message': 'No inspections performed yet'}
        
        latest = self.inspection_history[-1]
        
        return {
            'total_inspections': len(self.inspection_history),
            'latest_inspection': latest.timestamp,
            'current_health_score': latest.system_health_score,
            'trend_analysis': {
                'health_trend': self._get_health_trend(),
                'volatility_trend': self._get_volatility_trend(),
                'contradiction_trend': self._get_contradiction_trend()
            },
            'insights_cached': len(self.insight_cache),
            'drift_indicators': len(latest.cognitive_drift_indicators)
        }
    
    def _get_health_trend(self) -> str:
        """Analyze health score trend."""
        if len(self.inspection_history) < 3:
            return "insufficient_data"
        
        recent_scores = [s.system_health_score for s in self.inspection_history[-3:]]
        if recent_scores[-1] > recent_scores[0] + 0.1:
            return "improving"
        elif recent_scores[-1] < recent_scores[0] - 0.1:
            return "declining"
        else:
            return "stable"
    
    def _get_volatility_trend(self) -> str:
        """Analyze volatility trend."""
        if len(self.inspection_history) < 3:
            return "insufficient_data"
        
        recent_volatilities = [s.average_volatility for s in self.inspection_history[-3:]]
        if recent_volatilities[-1] < recent_volatilities[0] - 0.1:
            return "decreasing"
        elif recent_volatilities[-1] > recent_volatilities[0] + 0.1:
            return "increasing"
        else:
            return "stable"
    
    def _get_contradiction_trend(self) -> str:
        """Analyze contradiction trend."""
        if len(self.inspection_history) < 3:
            return "insufficient_data"
        
        recent_contradictions = [s.unresolved_contradictions for s in self.inspection_history[-3:]]
        if recent_contradictions[-1] < recent_contradictions[0] - 2:
            return "decreasing"
        elif recent_contradictions[-1] > recent_contradictions[0] + 2:
            return "increasing"
        else:
            return "stable" 