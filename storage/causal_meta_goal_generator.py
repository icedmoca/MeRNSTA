#!/usr/bin/env python3
"""
🧠 Causal-Aware Meta-Goal Generator for MeRNSTA

Analyzes causal links to generate intelligent meta-goals for:
- Clarifying weak causal connections
- Reinforcing strong patterns
- Identifying knowledge gaps in causal chains
- Auto-executing improvement queries
"""

import logging
import time
import uuid
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import math
from collections import defaultdict, Counter

from .enhanced_memory_model import EnhancedTripletFact
from .meta_cognition_agent import MetaGoal, MetaCognitionAgent
from .memory_log import MemoryLog


@dataclass
class CausalGoal(MetaGoal):
    """Extended meta-goal with causal analysis information."""
    causal_strength: float = 0.0
    causal_pattern: str = ""
    source_fact_id: Optional[int] = None
    target_fact_id: Optional[int] = None
    improvement_type: str = "clarification"  # "clarification", "reinforcement", "gap_filling", "validation"


class CausalMetaGoalGenerator:
    """
    Advanced causal-aware meta-goal generation system that analyzes
    causal patterns and generates targeted improvement strategies.
    """
    
    def __init__(self, memory_log: MemoryLog = None):
        self.memory_log = memory_log or MemoryLog()
        self.meta_cognition = MetaCognitionAgent()
        self.causal_goals: Dict[str, CausalGoal] = {}
        self.execution_queue: List[str] = []
        self.feedback_history: List[Dict] = []
        
        # Initialize enhanced memory system for proper fact retrieval
        try:
            from .enhanced_memory_system import EnhancedMemorySystem
            self.enhanced_memory = EnhancedMemorySystem()
        except ImportError:
            print("[CausalMetaGoal] Enhanced memory system not available, using memory_log")
            self.enhanced_memory = None
        
        # Thresholds for causal analysis
        self.weak_causal_threshold = 0.3
        self.strong_causal_threshold = 0.7
        self.volatility_threshold = 0.6
        self.confidence_threshold = 0.5
        
        print("[CausalMetaGoal] Initialized causal-aware meta-goal generator")

    def generate_causal_meta_goals(self, user_profile_id: str = None, 
                                 session_id: str = None) -> List[CausalGoal]:
        """
        Generate meta-goals based on causal link analysis.
        """
        print("[CausalMetaGoal] 🧠 Generating causal-aware meta-goals...")
        
        # Get recent facts with causal information
        facts = self._get_facts_with_causal_info(user_profile_id, session_id)
        causal_facts = [f for f in facts if hasattr(f, 'causal_strength') and f.causal_strength > 0]
        
        print(f"[CausalMetaGoal] Analyzing {len(causal_facts)} facts with causal information")
        
        goals = []
        
        # 1. Identify weak causal links that need clarification
        weak_goals = self._generate_weak_causal_goals(causal_facts)
        goals.extend(weak_goals)
        
        # 2. Identify strong patterns worth reinforcing
        strong_goals = self._generate_reinforcement_goals(causal_facts)
        goals.extend(strong_goals)
        
        # 3. Find causal chain gaps
        gap_goals = self._generate_gap_filling_goals(causal_facts)
        goals.extend(gap_goals)
        
        # 4. Validate old causal assumptions
        validation_goals = self._generate_validation_goals(causal_facts)
        goals.extend(validation_goals)
        
        # Store generated goals
        for goal in goals:
            self.causal_goals[goal.goal_id] = goal
            
        print(f"[CausalMetaGoal] Generated {len(goals)} causal meta-goals")
        return goals

    def _generate_weak_causal_goals(self, causal_facts: List[EnhancedTripletFact]) -> List[CausalGoal]:
        """Generate goals to clarify weak causal connections."""
        goals = []
        
        weak_facts = [f for f in causal_facts 
                     if f.causal_strength < self.weak_causal_threshold]
        
        # Group by causal patterns
        pattern_groups = defaultdict(list)
        for fact in weak_facts:
            if hasattr(fact, 'cause') and fact.cause:
                pattern = f"{fact.subject}_{fact.predicate}"
                pattern_groups[pattern].append(fact)
        
        for pattern, facts in pattern_groups.items():
            if len(facts) >= 2:  # Multiple weak instances of same pattern
                avg_strength = sum(f.causal_strength for f in facts) / len(facts)
                avg_confidence = sum(f.confidence for f in facts) / len(facts)
                
                goal = CausalGoal(
                    goal_id=f"weak_causal_{uuid.uuid4().hex[:8]}",
                    goal_type="clarification",
                    priority=7 if avg_confidence < 0.5 else 5,
                    description=f"Clarify weak causal pattern: {pattern.replace('_', ' ')}",
                    target_concept=pattern,
                    suggested_actions=[
                        f"Ask: 'Can you explain more about when {pattern.replace('_', ' ')}?'",
                        f"Ask: 'What typically causes you to {facts[0].predicate}?'",
                        f"Ask: 'How strong is the connection between these events?'"
                    ],
                    related_facts=facts,
                    created_time=time.time(),
                    causal_strength=avg_strength,
                    causal_pattern=pattern,
                    improvement_type="clarification"
                )
                goals.append(goal)
        
        return goals

    def _generate_reinforcement_goals(self, causal_facts: List[EnhancedTripletFact]) -> List[CausalGoal]:
        """Generate goals to reinforce strong causal patterns."""
        goals = []
        
        strong_facts = [f for f in causal_facts 
                       if f.causal_strength > self.strong_causal_threshold]
        
        # Look for patterns worth reinforcing
        pattern_groups = defaultdict(list)
        for fact in strong_facts:
            if hasattr(fact, 'cause') and fact.cause:
                # Create pattern from cause -> effect
                cause_parts = fact.cause.split()
                effect_pattern = f"{fact.subject}_{fact.predicate}"
                if len(cause_parts) >= 2:
                    cause_pattern = f"{cause_parts[0]}_{cause_parts[1]}"
                    full_pattern = f"{cause_pattern}_to_{effect_pattern}"
                    pattern_groups[full_pattern].append(fact)
        
        for pattern, facts in pattern_groups.items():
            if len(facts) >= 3:  # Strong pattern with multiple instances
                avg_strength = sum(f.causal_strength for f in facts) / len(facts)
                
                goal = CausalGoal(
                    goal_id=f"reinforce_{uuid.uuid4().hex[:8]}",
                    goal_type="consolidation", 
                    priority=4,
                    description=f"Reinforce strong causal pattern: {pattern.replace('_', ' ')}",
                    target_concept=pattern,
                    suggested_actions=[
                        f"Ask: 'Is this pattern always true for you?'",
                        f"Ask: 'Are there exceptions to this pattern?'",
                        f"Validate: 'This seems to be a consistent pattern in your experience'"
                    ],
                    related_facts=facts,
                    created_time=time.time(),
                    causal_strength=avg_strength,
                    causal_pattern=pattern,
                    improvement_type="reinforcement"
                )
                goals.append(goal)
        
        return goals

    def _generate_gap_filling_goals(self, causal_facts: List[EnhancedTripletFact]) -> List[CausalGoal]:
        """Generate goals to fill gaps in causal chains."""
        goals = []
        
        # Find subjects with missing causal connections
        subject_predicates = defaultdict(set)
        causal_subjects = set()
        
        for fact in causal_facts:
            subject_predicates[fact.subject].add(fact.predicate)
            causal_subjects.add(fact.subject)
        
        # Look for subjects with only effects but no causes, or vice versa
        for subject in causal_subjects:
            predicates = subject_predicates[subject]
            
            # Check if we have emotional states without clear causes
            emotional_predicates = {'feel', 'be'} & predicates
            action_predicates = {'work', 'study', 'exercise', 'eat', 'sleep'} & predicates
            
            if emotional_predicates and not action_predicates:
                goal = CausalGoal(
                    goal_id=f"gap_cause_{uuid.uuid4().hex[:8]}",
                    goal_type="exploration",
                    priority=6,
                    description=f"Explore missing causes for {subject}'s emotional states",
                    target_concept=f"{subject}_emotional_causes",
                    suggested_actions=[
                        f"Ask: 'What usually causes you to feel this way?'",
                        f"Ask: 'What activities or events lead to these feelings?'",
                        f"Explore: 'Tell me about what happened before you felt {list(emotional_predicates)[0]}'"
                    ],
                    related_facts=[f for f in causal_facts if f.subject == subject and f.predicate in emotional_predicates],
                    created_time=time.time(),
                    improvement_type="gap_filling"
                )
                goals.append(goal)
                
            elif action_predicates and not emotional_predicates:
                goal = CausalGoal(
                    goal_id=f"gap_effect_{uuid.uuid4().hex[:8]}",
                    goal_type="exploration", 
                    priority=5,
                    description=f"Explore missing effects of {subject}'s actions",
                    target_concept=f"{subject}_action_effects",
                    suggested_actions=[
                        f"Ask: 'How do you feel after {list(action_predicates)[0]}ing?'",
                        f"Ask: 'What are the effects of these activities on you?'",
                        f"Explore: 'Tell me about the impact of these activities'"
                    ],
                    related_facts=[f for f in causal_facts if f.subject == subject and f.predicate in action_predicates],
                    created_time=time.time(),
                    improvement_type="gap_filling"
                )
                goals.append(goal)
        
        return goals

    def _generate_validation_goals(self, causal_facts: List[EnhancedTripletFact]) -> List[CausalGoal]:
        """Generate goals to validate old causal assumptions."""
        goals = []
        
        current_time = time.time()
        old_facts = [f for f in causal_facts 
                    if hasattr(f, 'timestamp') and f.timestamp and 
                    (current_time - f.timestamp) > 7 * 24 * 3600]  # Older than 7 days
        
        if old_facts:
            # Group by causal strength
            medium_strength_facts = [f for f in old_facts 
                                   if self.weak_causal_threshold <= f.causal_strength <= self.strong_causal_threshold]
            
            if medium_strength_facts:
                goal = CausalGoal(
                    goal_id=f"validate_{uuid.uuid4().hex[:8]}",
                    goal_type="validation",
                    priority=3,
                    description="Validate older causal assumptions",
                    target_concept="historical_patterns",
                    suggested_actions=[
                        "Ask: 'Are these patterns still true for you?'",
                        "Ask: 'Have your habits or reactions changed recently?'", 
                        "Validate: 'Let me check if these connections still apply'"
                    ],
                    related_facts=medium_strength_facts[:5],  # Top 5 for validation
                    created_time=time.time(),
                    improvement_type="validation"
                )
                goals.append(goal)
        
        return goals

    def _get_facts_with_causal_info(self, user_profile_id: str = None, 
                                  session_id: str = None) -> List[EnhancedTripletFact]:
        """Get facts that have causal information."""
        try:
            # Use enhanced memory system if available, otherwise fall back to memory_log
            if self.enhanced_memory:
                all_facts = self.enhanced_memory.get_facts(user_profile_id=user_profile_id, session_id=session_id)
            else:
                all_facts = self.memory_log.get_all_facts()
            
            # Filter for facts with causal information - check multiple possible attributes
            causal_facts = []
            for fact in all_facts:
                has_causal_info = (
                    (hasattr(fact, 'causal_strength') and fact.causal_strength and fact.causal_strength >= 0.1) or
                    (hasattr(fact, 'cause') and fact.cause) or
                    (hasattr(fact, 'causal_link') and fact.causal_link)
                )
                if has_causal_info:
                    # Ensure causal_strength exists and has a reasonable value
                    if not hasattr(fact, 'causal_strength') or not fact.causal_strength:
                        fact.causal_strength = 0.3  # Default reasonable strength
                    causal_facts.append(fact)
            
            print(f"[CausalMetaGoal] Found {len(causal_facts)} facts with causal information out of {len(all_facts)} total facts")
            return causal_facts
            
        except Exception as e:
            print(f"[CausalMetaGoal] Error getting causal facts: {e}")
            # Fallback: try to get any recent facts
            try:
                if self.enhanced_memory:
                    recent_facts = self.enhanced_memory.get_facts(user_profile_id=user_profile_id, limit=10)
                else:
                    recent_facts = self.memory_log.get_all_facts()[-10:]  # Last 10 facts
                print(f"[CausalMetaGoal] Fallback: Using {len(recent_facts)} recent facts")
                return recent_facts
            except:
                return []

    def get_priority_causal_goals(self, limit: int = 3) -> List[CausalGoal]:
        """Get the highest priority causal goals for execution."""
        pending_goals = [g for g in self.causal_goals.values() if g.status == "pending"]
        pending_goals.sort(key=lambda x: x.priority, reverse=True)
        return pending_goals[:limit]

    def generate_execution_queries(self, goals: List[CausalGoal] = None) -> List[str]:
        """Generate specific queries to execute for causal improvement."""
        if goals is None:
            goals = self.get_priority_causal_goals()
        
        queries = []
        for goal in goals:
            # Select the most appropriate suggested action
            if goal.suggested_actions:
                query = goal.suggested_actions[0]  # Use first suggestion
                queries.append(query)
                
                # Add to execution queue for tracking
                self.execution_queue.append(goal.goal_id)
        
        return queries

    def process_user_feedback(self, goal_id: str, user_response: str, 
                            confidence_change: float = 0.0) -> Dict[str, Any]:
        """
        Process user feedback for a causal goal and update memory accordingly.
        """
        if goal_id not in self.causal_goals:
            return {"error": "Goal not found"}
        
        goal = self.causal_goals[goal_id]
        feedback_result = {
            "goal_id": goal_id,
            "goal_type": goal.goal_type,
            "user_response": user_response,
            "confidence_change": confidence_change,
            "actions_taken": []
        }
        
        # Analyze user response for positive/negative feedback
        positive_indicators = ["yes", "correct", "true", "always", "definitely", "absolutely"]
        negative_indicators = ["no", "wrong", "false", "never", "not", "incorrect"]
        
        user_lower = user_response.lower()
        is_positive = any(indicator in user_lower for indicator in positive_indicators)
        is_negative = any(indicator in user_lower for indicator in negative_indicators)
        
        # Update related facts based on feedback
        for fact in goal.related_facts:
            if is_positive:
                # Reinforce the causal link
                if hasattr(fact, 'causal_strength'):
                    fact.causal_strength = min(1.0, fact.causal_strength + 0.2)
                fact.confidence = min(1.0, fact.confidence + 0.1)
                feedback_result["actions_taken"].append(f"Reinforced causal strength for fact {fact.id}")
                
            elif is_negative:
                # Weaken or remove the causal link
                if hasattr(fact, 'causal_strength'):
                    fact.causal_strength = max(0.0, fact.causal_strength - 0.3)
                fact.confidence = max(0.1, fact.confidence - 0.2)
                feedback_result["actions_taken"].append(f"Weakened causal strength for fact {fact.id}")
        
        # Mark goal as completed
        goal.status = "completed"
        
        # Store feedback for learning
        self.feedback_history.append({
            "timestamp": time.time(),
            "goal_id": goal_id,
            "goal_type": goal.goal_type,
            "response": user_response,
            "is_positive": is_positive,
            "is_negative": is_negative,
            "confidence_change": confidence_change
        })
        
        print(f"[CausalMetaGoal] Processed feedback for goal {goal_id}: {len(feedback_result['actions_taken'])} actions taken")
        
        return feedback_result

    def get_causal_health_summary(self) -> Dict[str, Any]:
        """Get a summary of causal reasoning health."""
        try:
            facts = self._get_facts_with_causal_info()
            causal_facts = [f for f in facts if hasattr(f, 'causal_strength') and f.causal_strength > 0]
            
            if not causal_facts:
                return {"status": "no_causal_data", "total_facts": len(facts)}
            
            # Calculate statistics
            strengths = [f.causal_strength for f in causal_facts]
            avg_strength = sum(strengths) / len(strengths)
            strong_links = len([s for s in strengths if s > self.strong_causal_threshold])
            weak_links = len([s for s in strengths if s < self.weak_causal_threshold])
            
            # Goal statistics
            pending_goals = [g for g in self.causal_goals.values() if g.status == "pending"]
            completed_goals = [g for g in self.causal_goals.values() if g.status == "completed"]
            
            return {
                "status": "healthy" if avg_strength > 0.5 else "needs_attention",
                "total_causal_facts": len(causal_facts),
                "average_causal_strength": avg_strength,
                "strong_links": strong_links,
                "weak_links": weak_links,
                "pending_goals": len(pending_goals),
                "completed_goals": len(completed_goals),
                "feedback_sessions": len(self.feedback_history),
                "execution_queue_size": len(self.execution_queue)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)} 