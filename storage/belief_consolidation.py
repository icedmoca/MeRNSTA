#!/usr/bin/env python3
"""
🧬 Belief Consolidation System for MeRNSTA

Once a belief stabilizes (e.g. user keeps liking Python over time), reinforce it and 
optionally prune stale contradictory facts. Creates self-improving memory that 
strengthens consistent patterns and weakens conflicting information.
"""

import logging
import time
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import math

from .enhanced_triplet_extractor import EnhancedTripletFact


@dataclass
class BeliefPattern:
    """Represents a pattern of consistent beliefs over time."""
    pattern_id: str
    subject: str
    predicate: str
    object: str
    support_facts: List[EnhancedTripletFact]  # Facts supporting this belief
    contradictory_facts: List[EnhancedTripletFact]  # Facts contradicting this belief
    stability_score: float  # How stable this belief is over time
    reinforcement_events: List[Dict]  # Times when belief was reinforced
    last_reinforced: float
    consolidation_level: str  # "emerging", "stable", "consolidated"


class BeliefConsolidationSystem:
    """
    Dynamically identifies stable belief patterns and consolidates them over time.
    Prunes contradictory facts that are inconsistent with established beliefs.
    """
    
    def __init__(self):
        from config.settings import DEFAULT_VALUES
        
        self.belief_patterns: Dict[str, BeliefPattern] = {}
        self.stability_threshold = DEFAULT_VALUES.get("stability_threshold", 0.7)  # When to consider a belief stable
        self.consolidation_threshold = DEFAULT_VALUES.get("consolidation_threshold", 0.85)  # When to fully consolidate
        self.pruning_threshold = DEFAULT_VALUES.get("pruning_threshold", 0.9)  # When to prune contradictory facts
        self.temporal_window = 30 * 24 * 3600  # 30 days in seconds
        
    def analyze_belief_stability(self, facts: List[EnhancedTripletFact]) -> Dict[str, BeliefPattern]:
        """
        Analyze all facts to identify stable belief patterns and consolidate them.
        """
        print(f"[BeliefConsolidation] Analyzing {len(facts)} facts for belief patterns")
        
        # Group facts by (subject, predicate, object) patterns
        fact_groups = self._group_facts_by_pattern(facts)
        
        # Analyze each pattern for stability
        for pattern_key, pattern_facts in fact_groups.items():
            subject, predicate, obj = pattern_key
            self._analyze_pattern_stability(subject, predicate, obj, pattern_facts, facts)
        
        # Update consolidation levels
        self._update_consolidation_levels()
        
        # Identify facts for pruning
        prunable_facts = self._identify_prunable_facts()
        
        print(f"[BeliefConsolidation] Found {len(self.belief_patterns)} belief patterns")
        print(f"[BeliefConsolidation] Identified {len(prunable_facts)} facts for potential pruning")
        
        return self.belief_patterns
    
    def _group_facts_by_pattern(self, facts: List[EnhancedTripletFact]) -> Dict[Tuple[str, str, str], List[EnhancedTripletFact]]:
        """Group facts by their semantic pattern (subject, predicate, object)."""
        patterns = defaultdict(list)
        
        for fact in facts:
            # Normalize for pattern matching
            subject = fact.subject.lower().strip()
            predicate = fact.predicate.lower().strip()
            obj = fact.object.lower().strip()
            
            pattern_key = (subject, predicate, obj)
            patterns[pattern_key].append(fact)
        
        return patterns
    
    def _analyze_pattern_stability(self, subject: str, predicate: str, obj: str, 
                                  pattern_facts: List[EnhancedTripletFact], 
                                  all_facts: List[EnhancedTripletFact]):
        """Analyze the stability of a specific belief pattern."""
        pattern_id = f"{subject}_{predicate}_{obj}".replace(" ", "_")
        
        # Find contradictory facts
        contradictory_facts = self._find_contradictory_facts(subject, predicate, obj, all_facts)
        
        # Calculate temporal consistency
        temporal_score = self._calculate_temporal_consistency(pattern_facts)
        
        # Calculate reinforcement strength
        reinforcement_score = self._calculate_reinforcement_strength(pattern_facts, contradictory_facts)
        
        # Calculate overall stability
        stability_score = (temporal_score * 0.4 + reinforcement_score * 0.6)
        
        # Create or update belief pattern
        if pattern_id in self.belief_patterns:
            belief = self.belief_patterns[pattern_id]
            belief.support_facts = pattern_facts
            belief.contradictory_facts = contradictory_facts
            belief.stability_score = stability_score
        else:
            belief = BeliefPattern(
                pattern_id=pattern_id,
                subject=subject,
                predicate=predicate,
                object=obj,
                support_facts=pattern_facts,
                contradictory_facts=contradictory_facts,
                stability_score=stability_score,
                reinforcement_events=[],
                last_reinforced=time.time(),
                consolidation_level="emerging"
            )
            self.belief_patterns[pattern_id] = belief
        
        # Track reinforcement if pattern is strong
        if len(pattern_facts) > 1 and stability_score > 0.6:
            reinforcement_event = {
                'timestamp': time.time(),
                'support_count': len(pattern_facts),
                'contradiction_count': len(contradictory_facts),
                'stability_score': stability_score
            }
            belief.reinforcement_events.append(reinforcement_event)
            belief.last_reinforced = time.time()
            
            print(f"[BeliefConsolidation] Reinforced belief: {subject} {predicate} {obj} (stability: {stability_score:.2f})")
    
    def _find_contradictory_facts(self, subject: str, predicate: str, obj: str, 
                                 all_facts: List[EnhancedTripletFact]) -> List[EnhancedTripletFact]:
        """Find facts that contradict the given pattern."""
        contradictory = []
        
        for fact in all_facts:
            if (fact.subject.lower() == subject and 
                fact.predicate.lower() != predicate and
                fact.object.lower() == obj):
                # Same subject and object, different predicate (e.g., "like" vs "hate")
                if self._predicates_are_contradictory(predicate, fact.predicate.lower()):
                    contradictory.append(fact)
            elif (fact.subject.lower() == subject and 
                  fact.predicate.lower() == predicate and
                  fact.object.lower() != obj):
                # Same subject and predicate, different object (e.g., "prefer coffee" vs "prefer tea")
                if self._is_preference_predicate(predicate):
                    contradictory.append(fact)
        
        return contradictory
    
    def _predicates_are_contradictory(self, pred1: str, pred2: str) -> bool:
        """Check if two predicates are contradictory."""
        contradictory_pairs = [
            ('like', 'hate'), ('like', 'dislike'),
            ('love', 'hate'), ('love', 'dislike'),
            ('enjoy', 'hate'), ('enjoy', 'dislike'),
            ('prefer', 'dislike'), ('want', 'reject')
        ]
        
        pred1_clean = pred1.lower().strip()
        pred2_clean = pred2.lower().strip()
        
        for pair in contradictory_pairs:
            if (pred1_clean in pair and pred2_clean in pair and pred1_clean != pred2_clean):
                return True
        
        return False
    
    def _is_preference_predicate(self, predicate: str) -> bool:
        """Check if predicate indicates a preference choice."""
        preference_predicates = {'prefer', 'choose', 'select', 'pick', 'favor'}
        return predicate.lower().strip() in preference_predicates
    
    def _calculate_temporal_consistency(self, facts: List[EnhancedTripletFact]) -> float:
        """Calculate how consistent beliefs are over time."""
        if len(facts) <= 1:
            return 0.5  # Neutral for single facts
        
        # Get timestamps
        timestamps = []
        for fact in facts:
            if hasattr(fact, 'timestamp') and fact.timestamp:
                timestamps.append(fact.timestamp)
            else:
                timestamps.append(time.time())  # Default to now
        
        timestamps.sort()
        
        if len(timestamps) < 2:
            return 0.5
        
        # Calculate temporal spread and consistency
        time_span = timestamps[-1] - timestamps[0]
        
        if time_span == 0:
            return 0.8  # All facts from same time = reasonably consistent
        
        # Facts spread over time indicates consistency
        time_score = min(1.0, time_span / (7 * 24 * 3600))  # Normalize by week
        
        # More facts over time = higher consistency
        fact_density = len(facts) / max(1, time_span / (24 * 3600))  # Facts per day
        density_score = min(1.0, fact_density / 2.0)  # Normalize
        
        return (time_score * 0.6 + density_score * 0.4)
    
    def _calculate_reinforcement_strength(self, supporting_facts: List[EnhancedTripletFact], 
                                        contradictory_facts: List[EnhancedTripletFact]) -> float:
        """Calculate how strongly a belief is reinforced vs contradicted."""
        support_count = len(supporting_facts)
        contradiction_count = len(contradictory_facts)
        
        if support_count == 0:
            return 0.0
        
        if contradiction_count == 0:
            # No contradictions = high reinforcement
            return min(1.0, support_count / 3.0)  # Cap at 1.0, max at 3 supports
        
        # Calculate ratio with diminishing returns for contradictions
        support_weight = support_count
        contradiction_weight = contradiction_count * 1.5  # Contradictions hurt more
        
        ratio = support_weight / (support_weight + contradiction_weight)
        
        # Boost for multiple supports
        multiple_support_bonus = min(0.3, (support_count - 1) * 0.1)
        
        return min(1.0, ratio + multiple_support_bonus)
    
    def _update_consolidation_levels(self):
        """Update consolidation levels based on stability scores."""
        for belief in self.belief_patterns.values():
            if belief.stability_score >= self.consolidation_threshold:
                belief.consolidation_level = "consolidated"
                print(f"[BeliefConsolidation] CONSOLIDATED belief: {belief.subject} {belief.predicate} {belief.object}")
            elif belief.stability_score >= self.stability_threshold:
                belief.consolidation_level = "stable"
                print(f"[BeliefConsolidation] STABLE belief: {belief.subject} {belief.predicate} {belief.object}")
            else:
                belief.consolidation_level = "emerging"
    
    def _identify_prunable_facts(self) -> List[EnhancedTripletFact]:
        """Identify contradictory facts that could be pruned."""
        prunable = []
        
        for belief in self.belief_patterns.values():
            if belief.stability_score >= self.pruning_threshold:
                # This belief is very stable, consider pruning its contradictions
                for contradictory_fact in belief.contradictory_facts:
                    # Only prune if the contradictory fact is old and weak
                    fact_age = time.time() - getattr(contradictory_fact, 'timestamp', time.time())
                    fact_confidence = getattr(contradictory_fact, 'confidence', 0.5)
                    
                    if fact_age > self.temporal_window and fact_confidence < 0.6:
                        prunable.append(contradictory_fact)
                        print(f"[BeliefConsolidation] Identified for pruning: {contradictory_fact.predicate} {contradictory_fact.object} (age: {fact_age/86400:.1f} days)")
        
        return prunable
    
    def get_consolidated_beliefs(self) -> List[BeliefPattern]:
        """Get beliefs that have been fully consolidated."""
        return [belief for belief in self.belief_patterns.values() 
                if belief.consolidation_level == "consolidated"]
    
    def get_stable_beliefs(self) -> List[BeliefPattern]:
        """Get beliefs that are stable but not yet consolidated."""
        return [belief for belief in self.belief_patterns.values() 
                if belief.consolidation_level == "stable"]
    
    def get_consolidation_summary(self) -> Dict:
        """Get a summary of belief consolidation status."""
        consolidated = self.get_consolidated_beliefs()
        stable = self.get_stable_beliefs()
        emerging = [b for b in self.belief_patterns.values() if b.consolidation_level == "emerging"]
        
        return {
            'total_patterns': len(self.belief_patterns),
            'consolidated_beliefs': len(consolidated),
            'stable_beliefs': len(stable),
            'emerging_beliefs': len(emerging),
            'consolidation_rate': len(consolidated) / max(1, len(self.belief_patterns)),
            'top_consolidated': [
                {
                    'pattern': f"{b.subject} {b.predicate} {b.object}",
                    'stability_score': b.stability_score,
                    'support_count': len(b.support_facts),
                    'reinforcements': len(b.reinforcement_events)
                }
                for b in sorted(consolidated, key=lambda x: x.stability_score, reverse=True)[:5]
            ]
        }
    
    def suggest_consolidation_actions(self) -> List[str]:
        """Suggest actions based on belief consolidation analysis."""
        suggestions = []
        
        consolidated = self.get_consolidated_beliefs()
        stable = self.get_stable_beliefs()
        
        if consolidated:
            suggestions.append(f"💎 {len(consolidated)} beliefs are now consolidated and reinforced")
            for belief in consolidated[:3]:  # Top 3
                suggestions.append(f"   → {belief.subject} {belief.predicate} {belief.object} (confidence: {belief.stability_score:.2f})")
        
        if stable:
            suggestions.append(f"⚖️ {len(stable)} beliefs are stable and approaching consolidation")
        
        # Check for conflicting patterns that need resolution
        high_conflict_beliefs = [b for b in self.belief_patterns.values() 
                               if len(b.contradictory_facts) > len(b.support_facts)]
        
        if high_conflict_beliefs:
            suggestions.append(f"⚠️ {len(high_conflict_beliefs)} belief patterns have more contradictions than support")
            suggestions.append("   → Consider asking clarifying questions about these topics")
        
        return suggestions if suggestions else ["All beliefs appear to be consolidating normally."]
    
    def reinforce_belief(self, subject: str, predicate: str, obj: str) -> bool:
        """Manually reinforce a specific belief pattern."""
        pattern_id = f"{subject}_{predicate}_{obj}".replace(" ", "_")
        
        if pattern_id in self.belief_patterns:
            belief = self.belief_patterns[pattern_id]
            
            # Add reinforcement event
            reinforcement_event = {
                'timestamp': time.time(),
                'type': 'manual_reinforcement',
                'previous_stability': belief.stability_score
            }
            belief.reinforcement_events.append(reinforcement_event)
            belief.last_reinforced = time.time()
            
            # Boost stability score
            belief.stability_score = min(1.0, belief.stability_score + 0.1)
            
            print(f"[BeliefConsolidation] Manually reinforced: {subject} {predicate} {obj}")
            return True
        
        return False 