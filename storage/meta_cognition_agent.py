#!/usr/bin/env python3
"""
🤖 Meta-Cognition Agent for MeRNSTA

Periodically scan memory for contradiction clusters and:
- Propose clarification questions
- Generate self-improvement meta-goals  
- Flag drifted belief clusters
- Monitor cognitive health and suggest interventions
"""

import logging
import time
import random
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
import math

from .enhanced_triplet_extractor import EnhancedTripletFact


@dataclass
class MetaGoal:
    """Represents a meta-cognitive goal for system improvement."""
    goal_id: str
    goal_type: str  # "clarification", "consolidation", "exploration", "validation"
    priority: int  # 1-10, higher is more urgent
    description: str
    target_concept: str
    suggested_actions: List[str]
    related_facts: List[EnhancedTripletFact]
    created_time: float
    status: str = "pending"  # "pending", "active", "completed", "obsolete"


@dataclass
class CognitiveHealth:
    """Assessment of the system's cognitive health."""
    overall_score: float  # 0-1, higher is better
    coherence_score: float  # How internally consistent beliefs are
    completeness_score: float  # How complete knowledge appears to be
    stability_score: float  # How stable beliefs are over time
    issues_detected: List[str]
    recommendations: List[str]
    last_assessment: float


class MetaCognitionAgent:
    """
    Advanced meta-cognitive system that monitors its own thinking patterns,
    identifies knowledge gaps, and generates self-improvement strategies.
    """
    
    def __init__(self):
        self.meta_goals: Dict[str, MetaGoal] = {}
        self.goal_counter = 0
        self.assessment_history: List[CognitiveHealth] = []
        self.scan_interval = 3600  # 1 hour in seconds
        self.last_scan = 0.0
        self.priority_thresholds = {
            'critical': 8,
            'high': 6,
            'medium': 4,
            'low': 2
        }
        
    def perform_cognitive_scan(self, facts: List[EnhancedTripletFact],
                             contradiction_clusters=None,
                             belief_patterns=None) -> CognitiveHealth:
        """
        Perform a comprehensive scan of cognitive state and identify areas for improvement.
        """
        print(f"[MetaCognition] Performing cognitive scan on {len(facts)} facts")
        
        # Assess cognitive health
        cognitive_health = self._assess_cognitive_health(facts, contradiction_clusters, belief_patterns)
        
        # Generate meta-goals based on assessment
        new_goals = self._generate_meta_goals(facts, contradiction_clusters, belief_patterns, cognitive_health)
        
        # Update meta-goal priorities
        self._update_goal_priorities()
        
        # Store assessment
        self.assessment_history.append(cognitive_health)
        self.last_scan = time.time()
        
        print(f"[MetaCognition] Generated {len(new_goals)} new meta-goals")
        print(f"[MetaCognition] Cognitive health score: {cognitive_health.overall_score:.2f}")
        
        return cognitive_health
    
    def _assess_cognitive_health(self, facts: List[EnhancedTripletFact],
                               contradiction_clusters=None,
                               belief_patterns=None) -> CognitiveHealth:
        """Assess overall cognitive health and coherence."""
        
        # 1. Coherence Assessment
        coherence_score = self._assess_coherence(facts, contradiction_clusters)
        
        # 2. Completeness Assessment  
        completeness_score = self._assess_completeness(facts)
        
        # 3. Stability Assessment
        stability_score = self._assess_stability(facts, belief_patterns)
        
        # 4. Overall Score
        overall_score = (coherence_score * 0.4 + completeness_score * 0.3 + stability_score * 0.3)
        
        # 5. Identify Issues
        issues = self._identify_cognitive_issues(facts, contradiction_clusters, belief_patterns)
        
        # 6. Generate Recommendations
        recommendations = self._generate_cognitive_recommendations(
            coherence_score, completeness_score, stability_score, issues
        )
        
        return CognitiveHealth(
            overall_score=overall_score,
            coherence_score=coherence_score,
            completeness_score=completeness_score,
            stability_score=stability_score,
            issues_detected=issues,
            recommendations=recommendations,
            last_assessment=time.time()
        )
    
    def _assess_coherence(self, facts: List[EnhancedTripletFact], contradiction_clusters=None) -> float:
        """Assess how internally coherent the belief system is."""
        if not facts:
            return 1.0
        
        # Count contradictory facts
        contradictory_facts = sum(1 for f in facts if getattr(f, 'contradiction', False))
        contradiction_ratio = contradictory_facts / len(facts)
        
        # Cluster-based coherence
        cluster_penalty = 0.0
        if contradiction_clusters:
            unstable_clusters = sum(1 for c in contradiction_clusters.values() 
                                  if c.volatility_score > 0.7)
            cluster_penalty = min(0.3, unstable_clusters / max(1, len(contradiction_clusters)))
        
        # Calculate coherence score
        coherence = 1.0 - (contradiction_ratio * 0.5) - cluster_penalty
        
        return max(0.0, min(1.0, coherence))
    
    def _assess_completeness(self, facts: List[EnhancedTripletFact]) -> float:
        """Assess how complete the knowledge base appears to be."""
        if not facts:
            return 0.0
        
        # Analyze knowledge domains
        domains = self._identify_knowledge_domains(facts)
        
        # Check for domain coverage depth
        domain_depths = {}
        for domain, domain_facts in domains.items():
            # More facts and diverse predicates indicate better coverage
            fact_count = len(domain_facts)
            predicate_diversity = len(set(f.predicate for f in domain_facts))
            
            domain_depths[domain] = min(1.0, (fact_count / 10) * (predicate_diversity / 5))
        
        # Overall completeness based on average domain depth
        if domain_depths:
            completeness = sum(domain_depths.values()) / len(domain_depths)
        else:
            completeness = 0.0
        
        return max(0.0, min(1.0, completeness))
    
    def _assess_stability(self, facts: List[EnhancedTripletFact], belief_patterns=None) -> float:
        """Assess how stable beliefs are over time."""
        if not facts:
            return 1.0
        
        # Count facts with volatility markers
        volatile_facts = sum(1 for f in facts if getattr(f, 'volatility_score', 0.0) > 0.5)
        volatility_ratio = volatile_facts / len(facts)
        
        # Belief pattern stability
        pattern_stability = 0.5  # Default
        if belief_patterns:
            stable_patterns = sum(1 for p in belief_patterns.values() 
                                if p.consolidation_level in ['stable', 'consolidated'])
            if belief_patterns:
                pattern_stability = stable_patterns / len(belief_patterns)
        
        # Combined stability score
        stability = (1.0 - volatility_ratio * 0.6) * 0.6 + pattern_stability * 0.4
        
        return max(0.0, min(1.0, stability))
    
    def _identify_knowledge_domains(self, facts: List[EnhancedTripletFact]) -> Dict[str, List[EnhancedTripletFact]]:
        """Identify different knowledge domains represented in facts."""
        domains = defaultdict(list)
        
        # Simple domain classification based on predicates and objects
        for fact in facts:
            predicate = fact.predicate.lower()
            obj = fact.object.lower()
            
            # Classify by domain
            if any(word in predicate for word in ['like', 'love', 'prefer', 'enjoy', 'hate']):
                if any(food_word in obj for food_word in ['food', 'pizza', 'pasta', 'cuisine']):
                    domains['food_preferences'].append(fact)
                elif any(activity in obj for activity in ['exercise', 'sport', 'hobby']):
                    domains['activity_preferences'].append(fact)
                else:
                    domains['general_preferences'].append(fact)
            elif any(word in predicate for word in ['work', 'job', 'career', 'study']):
                domains['professional'].append(fact)
            elif any(word in predicate for word in ['live', 'visit', 'travel']):
                domains['location'].append(fact)
            elif any(word in predicate for word in ['believe', 'think', 'feel']):
                domains['beliefs_opinions'].append(fact)
            else:
                domains['general_knowledge'].append(fact)
        
        return domains
    
    def _identify_cognitive_issues(self, facts: List[EnhancedTripletFact],
                                 contradiction_clusters=None, 
                                 belief_patterns=None) -> List[str]:
        """Identify specific cognitive issues that need attention."""
        issues = []
        
        # High contradiction rate
        if facts:
            contradictory_facts = sum(1 for f in facts if getattr(f, 'contradiction', False))
            if contradictory_facts / len(facts) > 0.2:
                issues.append("High contradiction rate detected in memory")
        
        # Unstable belief clusters
        if contradiction_clusters:
            unstable_clusters = [c for c in contradiction_clusters.values() if c.volatility_score > 0.7]
            if len(unstable_clusters) > 2:
                issues.append(f"Multiple unstable belief clusters ({len(unstable_clusters)} found)")
        
        # Sparse knowledge domains
        domains = self._identify_knowledge_domains(facts)
        sparse_domains = [d for d, facts_list in domains.items() if len(facts_list) < 3]
        if len(sparse_domains) > len(domains) / 2:
            issues.append("Many knowledge domains have sparse coverage")
        
        # Lack of consolidated beliefs
        if belief_patterns:
            consolidated = sum(1 for p in belief_patterns.values() 
                             if p.consolidation_level == 'consolidated')
            if consolidated / len(belief_patterns) < 0.3:
                issues.append("Low rate of belief consolidation")
        
        # Old, unresolved contradictions
        old_contradictions = [f for f in facts 
                            if getattr(f, 'contradiction', False) and
                            hasattr(f, 'timestamp') and f.timestamp and
                            time.time() - f.timestamp > 7 * 24 * 3600]  # 7 days old
        
        if len(old_contradictions) > 5:
            issues.append(f"Multiple old unresolved contradictions ({len(old_contradictions)} found)")
        
        return issues
    
    def _generate_cognitive_recommendations(self, coherence: float, completeness: float,
                                          stability: float, issues: List[str]) -> List[str]:
        """Generate recommendations for improving cognitive health."""
        recommendations = []
        
        # Coherence improvements
        if coherence < 0.6:
            recommendations.append("Focus on resolving contradictions through clarification questions")
            recommendations.append("Consider belief consolidation for frequently contradicted topics")
        
        # Completeness improvements
        if completeness < 0.5:
            recommendations.append("Explore knowledge gaps through targeted questioning")
            recommendations.append("Encourage elaboration on sparsely covered topics")
        
        # Stability improvements
        if stability < 0.6:
            recommendations.append("Monitor volatile topics for stabilization opportunities")
            recommendations.append("Reinforce stable beliefs through consistent experiences")
        
        # Issue-specific recommendations
        for issue in issues:
            if "contradiction rate" in issue:
                recommendations.append("Implement more aggressive contradiction resolution")
            elif "unstable belief clusters" in issue:
                recommendations.append("Generate clarification questions for unstable clusters")
            elif "sparse coverage" in issue:
                recommendations.append("Ask follow-up questions to expand knowledge domains")
        
        return recommendations
    
    def _generate_meta_goals(self, facts: List[EnhancedTripletFact],
                           contradiction_clusters=None,
                           belief_patterns=None,
                           cognitive_health=None) -> List[MetaGoal]:
        """Generate specific meta-goals for cognitive improvement."""
        new_goals = []
        
        # 1. Clarification goals for unstable clusters
        if contradiction_clusters:
            for cluster_id, cluster in contradiction_clusters.items():
                if cluster.volatility_score > 0.7:
                    goal = self._create_clarification_goal(cluster)
                    new_goals.append(goal)
        
        # 2. Consolidation goals for stable patterns
        if belief_patterns:
            stable_patterns = [p for p in belief_patterns.values() 
                             if p.consolidation_level == 'stable']
            for pattern in stable_patterns[:3]:  # Top 3 stable patterns
                goal = self._create_consolidation_goal(pattern)
                new_goals.append(goal)
        
        # 3. Exploration goals for sparse domains
        domains = self._identify_knowledge_domains(facts)
        sparse_domains = [(d, facts_list) for d, facts_list in domains.items() if len(facts_list) < 3]
        for domain_name, domain_facts in sparse_domains[:2]:  # Top 2 sparse domains
            goal = self._create_exploration_goal(domain_name, domain_facts)
            new_goals.append(goal)
        
        # 4. Validation goals for old contradictions
        old_contradictions = [f for f in facts 
                            if getattr(f, 'contradiction', False) and
                            hasattr(f, 'timestamp') and f.timestamp and
                            time.time() - f.timestamp > 7 * 24 * 3600]
        
        if old_contradictions:
            goal = self._create_validation_goal(old_contradictions)
            new_goals.append(goal)
        
        # Store new goals
        for goal in new_goals:
            self.meta_goals[goal.goal_id] = goal
        
        return new_goals
    
    def _create_clarification_goal(self, cluster) -> MetaGoal:
        """Create a clarification goal for an unstable cluster."""
        self.goal_counter += 1
        goal_id = f"clarify_{self.goal_counter}"
        
        priority = min(10, max(1, int(cluster.volatility_score * 10)))
        
        # Generate specific clarification questions
        theme = cluster.concept_theme
        fact_objects = [f.object for f in cluster.contradictory_facts]
        
        actions = [
            f"Ask: 'Can you clarify your feelings about {theme}?'",
            f"Ask: 'I notice conflicting information about {', '.join(fact_objects[:3])}. Which is more accurate?'",
            f"Ask: 'Has your opinion about {theme} changed recently?'"
        ]
        
        return MetaGoal(
            goal_id=goal_id,
            goal_type="clarification",
            priority=priority,
            description=f"Clarify contradictions in {theme}",
            target_concept=theme,
            suggested_actions=actions,
            related_facts=cluster.contradictory_facts,
            created_time=time.time()
        )
    
    def _create_consolidation_goal(self, belief_pattern) -> MetaGoal:
        """Create a consolidation goal for a stable belief pattern."""
        self.goal_counter += 1
        goal_id = f"consolidate_{self.goal_counter}"
        
        priority = max(3, int(belief_pattern.stability_score * 7))
        
        actions = [
            f"Reinforce: '{belief_pattern.subject} {belief_pattern.predicate} {belief_pattern.object}'",
            f"Ask: 'This seems to be a stable preference - is this still accurate?'",
            f"Consider pruning contradictory facts older than 30 days"
        ]
        
        return MetaGoal(
            goal_id=goal_id,
            goal_type="consolidation",
            priority=priority,
            description=f"Consolidate stable belief: {belief_pattern.subject} {belief_pattern.predicate} {belief_pattern.object}",
            target_concept=f"{belief_pattern.predicate}_{belief_pattern.object}",
            suggested_actions=actions,
            related_facts=belief_pattern.support_facts,
            created_time=time.time()
        )
    
    def _create_exploration_goal(self, domain_name: str, domain_facts: List[EnhancedTripletFact]) -> MetaGoal:
        """Create an exploration goal for sparse knowledge domains."""
        self.goal_counter += 1
        goal_id = f"explore_{self.goal_counter}"
        
        priority = 5  # Medium priority for exploration
        
        # Generate exploration questions based on domain
        if domain_name == 'food_preferences':
            actions = [
                "Ask: 'What are some of your favorite foods?'",
                "Ask: 'Are there any cuisines you particularly enjoy or avoid?'",
                "Ask: 'Do you have any dietary preferences or restrictions?'"
            ]
        elif domain_name == 'activity_preferences':
            actions = [
                "Ask: 'What activities do you enjoy in your free time?'",
                "Ask: 'What sports or exercises do you like?'",
                "Ask: 'What hobbies are you passionate about?'"
            ]
        else:
            actions = [
                f"Ask follow-up questions about {domain_name}",
                f"Encourage elaboration on topics in {domain_name}",
                f"Explore connections within {domain_name}"
            ]
        
        return MetaGoal(
            goal_id=goal_id,
            goal_type="exploration",
            priority=priority,
            description=f"Expand knowledge in {domain_name}",
            target_concept=domain_name,
            suggested_actions=actions,
            related_facts=domain_facts,
            created_time=time.time()
        )
    
    def _create_validation_goal(self, old_contradictions: List[EnhancedTripletFact]) -> MetaGoal:
        """Create a validation goal for old contradictions."""
        self.goal_counter += 1
        goal_id = f"validate_{self.goal_counter}"
        
        priority = 7  # High priority for resolving old issues
        
        actions = [
            "Ask: 'I have some old conflicting information. Can we review and update it?'",
            "Present contradictions and ask for current preference",
            "Consider archiving very old contradictory facts"
        ]
        
        return MetaGoal(
            goal_id=goal_id,
            goal_type="validation",
            priority=priority,
            description=f"Validate {len(old_contradictions)} old contradictions",
            target_concept="contradiction_resolution",
            suggested_actions=actions,
            related_facts=old_contradictions,
            created_time=time.time()
        )
    
    def _update_goal_priorities(self):
        """Update priorities of existing goals based on current state."""
        current_time = time.time()
        
        for goal in self.meta_goals.values():
            # Increase priority of older goals
            age_days = (current_time - goal.created_time) / (24 * 3600)
            age_bonus = min(3, age_days / 7)  # +1 priority per week, max +3
            
            # Decrease priority if goal type is overrepresented
            type_count = sum(1 for g in self.meta_goals.values() if g.goal_type == goal.goal_type)
            if type_count > 3:
                type_penalty = min(2, type_count - 3)
            else:
                type_penalty = 0
            
            # Update priority
            new_priority = goal.priority + age_bonus - type_penalty
            goal.priority = max(1, min(10, int(new_priority)))
    
    def get_priority_goals(self, limit: int = 5) -> List[MetaGoal]:
        """Get the highest priority meta-goals."""
        active_goals = [g for g in self.meta_goals.values() if g.status == "pending"]
        active_goals.sort(key=lambda x: x.priority, reverse=True)
        return active_goals[:limit]
    
    def generate_self_improvement_suggestions(self) -> List[str]:
        """Generate high-level self-improvement suggestions."""
        suggestions = []
        
        priority_goals = self.get_priority_goals(3)
        
        if not priority_goals:
            return ["Cognitive system appears to be functioning well. Continue monitoring."]
        
        for goal in priority_goals:
            if goal.goal_type == "clarification":
                suggestions.append(f"🔍 CLARIFICATION NEEDED: {goal.description}")
                suggestions.extend([f"   → {action}" for action in goal.suggested_actions[:2]])
            
            elif goal.goal_type == "consolidation":
                suggestions.append(f"💎 CONSOLIDATION OPPORTUNITY: {goal.description}")
                suggestions.extend([f"   → {action}" for action in goal.suggested_actions[:2]])
            
            elif goal.goal_type == "exploration":
                suggestions.append(f"🌟 EXPLORATION SUGGESTED: {goal.description}")
                suggestions.extend([f"   → {action}" for action in goal.suggested_actions[:2]])
            
            elif goal.goal_type == "validation":
                suggestions.append(f"✅ VALIDATION REQUIRED: {goal.description}")
                suggestions.extend([f"   → {action}" for action in goal.suggested_actions[:2]])
        
        return suggestions
    
    def mark_goal_completed(self, goal_id: str):
        """Mark a meta-goal as completed."""
        if goal_id in self.meta_goals:
            self.meta_goals[goal_id].status = "completed"
            print(f"[MetaCognition] Goal {goal_id} marked as completed")
    
    def get_cognitive_summary(self) -> Dict:
        """Get a summary of cognitive state and meta-goals."""
        if self.assessment_history:
            latest_health = self.assessment_history[-1]
        else:
            latest_health = None
        
        goal_counts = Counter(g.goal_type for g in self.meta_goals.values() if g.status == "pending")
        priority_counts = Counter()
        
        for goal in self.meta_goals.values():
            if goal.status == "pending":
                if goal.priority >= self.priority_thresholds['critical']:
                    priority_counts['critical'] += 1
                elif goal.priority >= self.priority_thresholds['high']:
                    priority_counts['high'] += 1
                elif goal.priority >= self.priority_thresholds['medium']:
                    priority_counts['medium'] += 1
                else:
                    priority_counts['low'] += 1
        
        return {
            'cognitive_health': {
                'overall_score': latest_health.overall_score if latest_health else 0.5,
                'coherence': latest_health.coherence_score if latest_health else 0.5,
                'completeness': latest_health.completeness_score if latest_health else 0.5,
                'stability': latest_health.stability_score if latest_health else 0.5,
                'issues_count': len(latest_health.issues_detected) if latest_health else 0
            },
            'meta_goals': {
                'total_active': len([g for g in self.meta_goals.values() if g.status == "pending"]),
                'by_type': dict(goal_counts),
                'by_priority': dict(priority_counts),
                'oldest_goal_age': max([(time.time() - g.created_time) / 86400 
                                      for g in self.meta_goals.values() if g.status == "pending"], 
                                     default=0.0)
            },
            'last_scan': self.last_scan,
            'scan_interval': self.scan_interval
        } 