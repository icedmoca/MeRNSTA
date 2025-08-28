#!/usr/bin/env python3
"""
🗣️ Dialogue Clarification Agent for MeRNSTA Phase 2

Auto-generates clarifying questions for high-volatility belief clusters.
Integrates with meta-cognition agent for periodic contradiction resolution.
Hooks into chat_ui for inline clarification requests.
"""

import logging
import time
import random
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
import re

from .enhanced_memory_model import EnhancedTripletFact


@dataclass
class ClarificationRequest:
    """Represents a clarification request to be presented to the user."""
    request_id: str
    question: str
    context: str  # Brief explanation of why clarification is needed
    priority: int  # 1-10, higher is more urgent
    related_facts: List[EnhancedTripletFact]
    volatility_cluster_id: Optional[str] = None
    created_time: float = 0.0
    status: str = "pending"  # "pending", "answered", "dismissed", "expired"
    attempts: int = 0  # How many times we've asked this


@dataclass
class ClarificationResponse:
    """User's response to a clarification request."""
    request_id: str
    user_response: str
    resolution_action: str  # "confirm", "update", "choose_one", "clarify_context"
    chosen_fact_id: Optional[str] = None  # If user chose one fact over another
    new_belief: Optional[str] = None  # If user provided updated belief
    timestamp: float = 0.0


class DialogueClarificationAgent:
    """
    Monitors belief volatility and generates targeted clarification questions
    to resolve contradictions and stabilize volatile beliefs.
    """
    
    def __init__(self):
        self.pending_requests: Dict[str, ClarificationRequest] = {}
        self.completed_requests: List[ClarificationResponse] = []
        self.request_counter = 0
        
        # Thresholds and configuration
        from config.settings import DEFAULT_VALUES
        self.volatility_threshold = DEFAULT_VALUES.get('clarification_volatility_threshold', 0.7)  # When to trigger clarification
        self.max_pending_requests = 5   # Don't overwhelm user
        self.request_cooldown = 3600     # 1 hour between similar requests
        self.max_attempts = 3            # Don't ask same thing too many times
        
        # Question templates for different types of contradictions
        self.question_templates = {
            "preference_conflict": [
                "I noticed you've expressed conflicting preferences about {topic}. You said you {statement1} but also {statement2}. Which better reflects your current feeling?",
                "You've given me conflicting information about {topic}. Do you {statement1} or {statement2}?",
                "I'm seeing mixed signals about {topic}. Could you clarify whether you {statement1} or {statement2}?"
            ],
            "belief_evolution": [
                "Your opinion about {topic} seems to have changed over time. You previously said {old_belief}, but recently mentioned {new_belief}. Has your view evolved?",
                "I have different information about {topic} from different times. Earlier you said {old_belief}, now {new_belief}. Which is more accurate now?",
                "Your beliefs about {topic} appear to have shifted. You used to say {old_belief}, but now {new_belief}. Is this change intentional?"
            ],
            "intensity_conflict": [
                "You've expressed different levels of feeling about {topic}. You said you {weak_statement} but also {strong_statement}. How strongly do you actually feel?",
                "I'm getting mixed intensity signals about {topic}. Do you {weak_statement} or {strong_statement}?",
                "There's inconsistency in how strongly you feel about {topic}. You mentioned both {weak_statement} and {strong_statement}. Which is more accurate?"
            ],
            "context_dependent": [
                "Your feelings about {topic} might depend on context. You said {statement1} and {statement2}. Could you help me understand when each applies?",
                "I think your views on {topic} might be situational. You've said {statement1} and {statement2}. Are both true in different contexts?",
                "Your beliefs about {topic} seem context-dependent. When you said {statement1} vs {statement2}, were you thinking of different situations?"
            ]
        }
    
    def analyze_volatility_and_generate_requests(self, facts: List[EnhancedTripletFact],
                                               contradiction_clusters: Dict = None) -> List[ClarificationRequest]:
        """
        Analyze facts for volatility and generate appropriate clarification requests.
        """
        print(f"[ClarificationAgent] Analyzing {len(facts)} facts for clarification opportunities")
        
        new_requests = []
        
        # Skip if we already have too many pending requests
        if len(self.pending_requests) >= self.max_pending_requests:
            print(f"[ClarificationAgent] Max pending requests reached ({self.max_pending_requests})")
            return new_requests
        
        # Process contradiction clusters first (higher priority)
        if contradiction_clusters:
            cluster_requests = self._generate_cluster_clarifications(contradiction_clusters)
            new_requests.extend(cluster_requests)
        
        # Process individual volatile facts
        volatile_requests = self._generate_volatile_fact_clarifications(facts)
        new_requests.extend(volatile_requests)
        
        # Process temporal belief evolution
        evolution_requests = self._generate_belief_evolution_clarifications(facts)
        new_requests.extend(evolution_requests)
        
        # Store pending requests
        for request in new_requests:
            self.pending_requests[request.request_id] = request
        
        print(f"[ClarificationAgent] Generated {len(new_requests)} new clarification requests")
        return new_requests
    
    def _generate_cluster_clarifications(self, contradiction_clusters: Dict) -> List[ClarificationRequest]:
        """Generate clarifications for contradiction clusters."""
        requests = []
        
        for cluster_id, cluster in contradiction_clusters.items():
            if cluster.volatility_score < self.volatility_threshold:
                continue
            
            # Check cooldown - don't ask about same cluster too often
            if self._is_on_cooldown(f"cluster_{cluster_id}"):
                continue
            
            cluster_facts = cluster.contradictory_facts
            if len(cluster_facts) < 2:
                continue
            
            # Determine the type of conflict
            conflict_type = self._classify_conflict_type(cluster_facts)
            
            # Generate appropriate question
            question = self._generate_cluster_question(cluster, conflict_type)
            
            if question:
                self.request_counter += 1
                request = ClarificationRequest(
                    request_id=f"cluster_{self.request_counter}",
                    question=question,
                    context=f"Detected conflicting beliefs about {cluster.concept_theme} (volatility: {cluster.volatility_score:.2f})",
                    priority=min(10, int(cluster.volatility_score * 10) + 2),  # Higher priority for volatile clusters
                    related_facts=cluster_facts,
                    volatility_cluster_id=cluster_id,
                    created_time=time.time()
                )
                requests.append(request)
                
                print(f"[ClarificationAgent] Generated cluster clarification for {cluster.concept_theme}")
        
        return requests
    
    def _generate_volatile_fact_clarifications(self, facts: List[EnhancedTripletFact]) -> List[ClarificationRequest]:
        """Generate clarifications for individual volatile facts."""
        requests = []
        
        # Group facts by (subject, object) to find conflicts
        fact_groups = defaultdict(list)
        for fact in facts:
            if fact.volatility_score >= self.volatility_threshold:
                key = (fact.subject.lower(), fact.object.lower())
                fact_groups[key].append(fact)
        
        for (subject, obj), group_facts in fact_groups.items():
            if len(group_facts) < 2:
                continue
            
            # Check cooldown
            if self._is_on_cooldown(f"volatile_{subject}_{obj}"):
                continue
            
            # Find conflicting predicates
            predicates = [f.predicate for f in group_facts]
            if len(set(predicates)) > 1:
                conflict_type = self._classify_conflict_type(group_facts)
                question = self._generate_volatile_question(group_facts, conflict_type)
                
                if question:
                    self.request_counter += 1
                    avg_volatility = sum(f.volatility_score for f in group_facts) / len(group_facts)
                    
                    request = ClarificationRequest(
                        request_id=f"volatile_{self.request_counter}",
                        question=question,
                        context=f"Multiple conflicting beliefs about {obj} (avg volatility: {avg_volatility:.2f})",
                        priority=max(1, int(avg_volatility * 8)),
                        related_facts=group_facts,
                        created_time=time.time()
                    )
                    requests.append(request)
        
        return requests
    
    def _generate_belief_evolution_clarifications(self, facts: List[EnhancedTripletFact]) -> List[ClarificationRequest]:
        """Generate clarifications for beliefs that have evolved over time."""
        requests = []
        
        # Group facts by (subject, predicate, object) pattern
        pattern_groups = defaultdict(list)
        for fact in facts:
            if fact.preceded_by:  # Only facts with temporal links
                pattern_key = (fact.subject.lower(), fact.predicate.lower())
                pattern_groups[pattern_key].append(fact)
        
        for pattern_key, group_facts in pattern_groups.items():
            if len(group_facts) < 2:
                continue
            
            # Sort by timestamp to see evolution
            sorted_facts = sorted(group_facts, key=lambda f: f.timestamp or 0)
            
            # Check if there's significant evolution
            if self._has_significant_evolution(sorted_facts):
                oldest_fact = sorted_facts[0]
                newest_fact = sorted_facts[-1]
                
                # Check cooldown
                if self._is_on_cooldown(f"evolution_{pattern_key[0]}_{pattern_key[1]}"):
                    continue
                
                question = self._generate_evolution_question(oldest_fact, newest_fact)
                
                if question:
                    self.request_counter += 1
                    
                    request = ClarificationRequest(
                        request_id=f"evolution_{self.request_counter}",
                        question=question,
                        context=f"Belief evolution detected over {newest_fact.get_age_days() - oldest_fact.get_age_days():.1f} days",
                        priority=6,  # Medium priority for evolution
                        related_facts=sorted_facts,
                        created_time=time.time()
                    )
                    requests.append(request)
        
        return requests
    
    def _classify_conflict_type(self, facts: List[EnhancedTripletFact]) -> str:
        """Classify the type of conflict between facts."""
        predicates = [f.predicate.lower() for f in facts]
        
        # Check for preference conflicts
        preference_predicates = {'like', 'love', 'hate', 'dislike', 'prefer', 'enjoy'}
        if any(p in preference_predicates for p in predicates):
            return "preference_conflict"
        
        # Check for intensity conflicts (same predicate, different objects or contexts)
        if len(set(predicates)) == 1:
            return "intensity_conflict"
        
        # Check for temporal evolution
        if any(f.preceded_by for f in facts):
            return "belief_evolution"
        
        # Default to context-dependent
        return "context_dependent"
    
    def _generate_cluster_question(self, cluster, conflict_type: str) -> str:
        """Generate a question for a contradiction cluster."""
        facts = cluster.contradictory_facts
        if len(facts) < 2:
            return ""
        
        templates = self.question_templates.get(conflict_type, self.question_templates["preference_conflict"])
        template = random.choice(templates)
        
        # Extract key information from facts
        topic = cluster.concept_theme.replace("_", " ")
        fact1, fact2 = facts[0], facts[1]
        
        statement1 = f"{fact1.predicate} {fact1.object}"
        statement2 = f"{fact2.predicate} {fact2.object}"
        
        try:
            question = template.format(
                topic=topic,
                statement1=statement1,
                statement2=statement2
            )
            return question
        except KeyError:
            # Fallback question
            return f"I have conflicting information about {topic}. You said you {statement1} but also {statement2}. Could you clarify which is more accurate?"
    
    def _generate_volatile_question(self, facts: List[EnhancedTripletFact], conflict_type: str) -> str:
        """Generate a question for volatile facts."""
        if len(facts) < 2:
            return ""
        
        fact1, fact2 = facts[0], facts[1]
        topic = fact1.object
        
        if conflict_type == "intensity_conflict":
            # Different predicates, same object
            weak_stmt = f"{fact1.predicate} {fact1.object}"
            strong_stmt = f"{fact2.predicate} {fact2.object}"
            
            # Determine which is stronger
            intensity_order = ['like', 'enjoy', 'love', 'hate', 'dislike']
            pred1_idx = intensity_order.index(fact1.predicate) if fact1.predicate in intensity_order else 0
            pred2_idx = intensity_order.index(fact2.predicate) if fact2.predicate in intensity_order else 0
            
            if pred1_idx > pred2_idx:
                weak_stmt, strong_stmt = strong_stmt, weak_stmt
            
            templates = self.question_templates["intensity_conflict"]
            template = random.choice(templates)
            
            return template.format(
                topic=topic,
                weak_statement=weak_stmt,
                strong_statement=strong_stmt
            )
        else:
            # Default preference conflict
            return f"You've given me conflicting information about {topic}. Do you {fact1.predicate} it or {fact2.predicate} it?"
    
    def _generate_evolution_question(self, old_fact: EnhancedTripletFact, new_fact: EnhancedTripletFact) -> str:
        """Generate a question about belief evolution."""
        templates = self.question_templates["belief_evolution"]
        template = random.choice(templates)
        
        topic = old_fact.object
        old_belief = f"{old_fact.predicate} {old_fact.object}"
        new_belief = f"{new_fact.predicate} {new_fact.object}"
        
        return template.format(
            topic=topic,
            old_belief=old_belief,
            new_belief=new_belief
        )
    
    def _has_significant_evolution(self, sorted_facts: List[EnhancedTripletFact]) -> bool:
        """Check if facts show significant belief evolution."""
        if len(sorted_facts) < 2:
            return False
        
        # Check if predicates are different
        predicates = [f.predicate for f in sorted_facts]
        if len(set(predicates)) < 2:
            return False
        
        # Check if time span is significant (at least 1 day)
        time_span = sorted_facts[-1].timestamp - sorted_facts[0].timestamp
        if time_span < 24 * 3600:  # Less than 1 day
            return False
        
        return True
    
    def _is_on_cooldown(self, request_type: str) -> bool:
        """Check if we should wait before asking about this topic again."""
        # Check recent requests for similar topics
        recent_cutoff = time.time() - self.request_cooldown
        
        for request in self.pending_requests.values():
            if (request.created_time > recent_cutoff and 
                request_type in request.request_id and
                request.attempts < self.max_attempts):
                return True
        
        return False
    
    def process_user_response(self, request_id: str, user_response: str) -> ClarificationResponse:
        """Process user's response to a clarification request."""
        if request_id not in self.pending_requests:
            raise ValueError(f"No pending request with ID: {request_id}")
        
        request = self.pending_requests[request_id]
        
        # Analyze user response to determine action
        resolution_action = self._analyze_user_response(user_response, request)
        
        response = ClarificationResponse(
            request_id=request_id,
            user_response=user_response,
            resolution_action=resolution_action,
            timestamp=time.time()
        )
        
        # Extract specific information based on resolution action
        if resolution_action == "choose_one":
            response.chosen_fact_id = self._extract_chosen_fact(user_response, request.related_facts)
        elif resolution_action == "update":
            response.new_belief = self._extract_new_belief(user_response)
        
        # Move request to completed
        self.completed_requests.append(response)
        request.status = "answered"
        del self.pending_requests[request_id]
        
        print(f"[ClarificationAgent] Processed response for request {request_id}: {resolution_action}")
        
        return response
    
    def _analyze_user_response(self, response: str, request: ClarificationRequest) -> str:
        """Analyze user response to determine the type of resolution."""
        response_lower = response.lower().strip()
        
        # Check for explicit choices
        if any(word in response_lower for word in ['first', 'second', 'former', 'latter', 'option 1', 'option 2']):
            return "choose_one"
        
        # Check for updates/corrections
        if any(word in response_lower for word in ['actually', 'now', 'changed', 'different', 'update']):
            return "update"
        
        # Check for confirmations
        if any(word in response_lower for word in ['yes', 'correct', 'right', 'true', 'confirm']):
            return "confirm"
        
        # Check for context clarification
        if any(word in response_lower for word in ['depends', 'context', 'situation', 'sometimes', 'when']):
            return "clarify_context"
        
        # Default to clarify_context if unclear
        return "clarify_context"
    
    def _extract_chosen_fact(self, response: str, facts: List[EnhancedTripletFact]) -> Optional[str]:
        """Extract which fact the user chose from their response."""
        response_lower = response.lower()
        
        # Look for explicit choices
        if 'first' in response_lower or 'option 1' in response_lower or 'former' in response_lower:
            return facts[0].id if facts else None
        elif 'second' in response_lower or 'option 2' in response_lower or 'latter' in response_lower:
            return facts[1].id if len(facts) > 1 else None
        
        # Look for fact content matches
        for fact in facts:
            if fact.predicate.lower() in response_lower and fact.object.lower() in response_lower:
                return fact.id
        
        return None
    
    def _extract_new_belief(self, response: str) -> Optional[str]:
        """Extract the new belief statement from user response."""
        # Look for patterns like "I actually..." or "Now I..."
        patterns = [
            r"actually (.+?)(?:\.|$)",
            r"now (?:i )?(.+?)(?:\.|$)",
            r"(?:my|the) (?:new|current) (?:belief|opinion) is (.+?)(?:\.|$)",
            r"i (?:now )?(?:think|believe|feel) (.+?)(?:\.|$)"
        ]
        
        response_lower = response.lower().strip()
        
        for pattern in patterns:
            match = re.search(pattern, response_lower)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, return the full response (cleaned up)
        return response.strip()
    
    def get_pending_requests_for_ui(self) -> List[Dict[str, Any]]:
        """Get pending requests formatted for UI display."""
        ui_requests = []
        
        for request in sorted(self.pending_requests.values(), key=lambda r: r.priority, reverse=True):
            ui_requests.append({
                'id': request.request_id,
                'question': request.question,
                'context': request.context,
                'priority': request.priority,
                'created_time': request.created_time,
                'attempts': request.attempts,
                'facts_count': len(request.related_facts)
            })
        
        return ui_requests
    
    def dismiss_request(self, request_id: str, reason: str = "user_dismissed"):
        """Dismiss a clarification request."""
        if request_id in self.pending_requests:
            request = self.pending_requests[request_id]
            request.status = "dismissed"
            del self.pending_requests[request_id]
            
            print(f"[ClarificationAgent] Dismissed request {request_id}: {reason}")
    
    def cleanup_expired_requests(self, max_age_hours: int = 24):
        """Remove old pending requests that haven't been answered."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        expired_ids = []
        
        for request_id, request in self.pending_requests.items():
            if request.created_time < cutoff_time:
                expired_ids.append(request_id)
        
        for request_id in expired_ids:
            self.pending_requests[request_id].status = "expired"
            del self.pending_requests[request_id]
        
        if expired_ids:
            print(f"[ClarificationAgent] Cleaned up {len(expired_ids)} expired requests")
    
    def get_clarification_summary(self) -> Dict[str, Any]:
        """Get summary of clarification activity."""
        return {
            'pending_requests': len(self.pending_requests),
            'completed_requests': len(self.completed_requests),
            'max_pending': self.max_pending_requests,
            'total_generated': self.request_counter,
            'success_rate': len([r for r in self.completed_requests if r.resolution_action != "clarify_context"]) / max(1, len(self.completed_requests)),
            'recent_activity': len([r for r in self.completed_requests if time.time() - r.timestamp < 3600])  # Last hour
        } 