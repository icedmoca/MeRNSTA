#!/usr/bin/env python3
"""
🧠 Theory of Mind Layer for MeRNSTA Phase 2

Supports perspective-tagged beliefs and third-party belief tracking:
- Store beliefs attributed to different agents/people
- Track nested beliefs ("user believes that Anna believes...")
- Detect contradictions across perspectives
- Identify potential deception or misalignment
- Support belief attribution and confidence by subject
"""

import logging
import time
import re
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import uuid

from .enhanced_memory_model import EnhancedTripletFact


@dataclass
class PerspectiveAgent:
    """Represents an agent/person whose beliefs we track."""
    agent_id: str
    name: str
    agent_type: str  # "user", "person", "system", "group"
    trust_level: float = 0.8  # How much we trust this agent's statements (0.0-1.0)
    consistency_score: float = 0.5  # How consistent this agent's beliefs are
    last_interaction: Optional[float] = None
    belief_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BeliefAttribution:
    """Represents who believes what, with confidence tracking."""
    fact_id: str
    belief_holder: str  # Who holds this belief
    attributed_by: str  # Who told us about this belief  
    confidence: float   # How confident the attribution is
    context: Optional[str] = None  # Context in which belief was expressed
    timestamp: float = 0.0


@dataclass
class NestedBelief:
    """Represents nested beliefs like 'user believes that Anna believes X'."""
    belief_id: str
    outer_agent: str    # Who believes that someone else believes something
    inner_agent: str    # Who the outer agent thinks believes something
    core_belief: EnhancedTripletFact  # The actual belief content
    nesting_depth: int  # How many levels deep this goes
    confidence_chain: List[float]  # Confidence at each level
    original_statement: str  # Original text that created this nested belief


class TheoryOfMindLayer:
    """
    Manages perspective-aware beliefs and tracks what different agents believe.
    Enables understanding of beliefs about beliefs and detection of deception.
    """
    
    def __init__(self):
        self.agents: Dict[str, PerspectiveAgent] = {}
        self.belief_attributions: Dict[str, List[BeliefAttribution]] = defaultdict(list)
        self.nested_beliefs: Dict[str, NestedBelief] = {}
        self.perspective_contradictions: List[Dict] = []
        
        # Initialize core agents
        self._initialize_core_agents()
        
        # Patterns for detecting belief attribution in text
        self.attribution_patterns = [
            # Direct attribution
            r"(\w+) (?:said|told me|mentioned|claimed|stated) (?:that )?(.+)",
            r"according to (\w+),? (.+)",
            r"(\w+) believes? (?:that )?(.+)",
            r"(\w+) thinks? (?:that )?(.+)",
            
            # Nested beliefs
            r"i (?:think|believe) (?:that )?(\w+) (?:thinks|believes) (?:that )?(.+)",
            r"(\w+) (?:told me|said) (?:that )?(\w+) (?:thinks|believes) (?:that )?(.+)",
            
            # Hearsay and rumors
            r"i heard (?:from (\w+) )?(?:that )?(.+)",
            r"someone (?:told me|said) (?:that )?(.+)",
            r"they say (?:that )?(.+)",
            
            # Doubt and uncertainty about others
            r"i doubt (?:that )?(\w+) (?:really )?(?:thinks|believes) (?:that )?(.+)",
            r"(\w+) probably (?:doesn't )?(?:think|believe) (?:that )?(.+)",
        ]
        
        print("[TheoryOfMind] Initialized with perspective tracking")
    
    def _initialize_core_agents(self):
        """Initialize core agents that we always track."""
        self.agents["user"] = PerspectiveAgent(
            agent_id="user",
            name="User",
            agent_type="user",
            trust_level=1.0,
            consistency_score=0.5
        )
        
        self.agents["system"] = PerspectiveAgent(
            agent_id="system", 
            name="System",
            agent_type="system",
            trust_level=1.0,
            consistency_score=0.9
        )
    
    def process_statement_for_attribution(self, statement: str, 
                                        speaker: str = "user") -> List[EnhancedTripletFact]:
        """
        Analyze a statement to extract belief attributions and create perspective-tagged facts.
        """
        print(f"[TheoryOfMind] Processing statement for attribution: '{statement}'")
        
        attributed_facts = []
        
        # Try to match attribution patterns
        for pattern in self.attribution_patterns:
            matches = re.findall(pattern, statement.lower(), re.IGNORECASE)
            
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    attributed_fact = self._create_attributed_fact(match, statement, speaker)
                    if attributed_fact:
                        attributed_facts.append(attributed_fact)
                        
        # If no attribution patterns match, treat as direct user belief
        if not attributed_facts and speaker == "user":
            # Check for nested belief indicators
            nested_fact = self._check_for_nested_belief(statement, speaker)
            if nested_fact:
                attributed_facts.append(nested_fact)
        
        return attributed_facts
    
    def _create_attributed_fact(self, match: Tuple, original_statement: str, 
                              speaker: str) -> Optional[EnhancedTripletFact]:
        """Create a fact with proper attribution from a pattern match."""
        if len(match) == 2:
            agent_name, belief_content = match
            agent_name = agent_name.strip().lower()
            belief_content = belief_content.strip()
            
            # Clean up agent name
            agent_name = self._normalize_agent_name(agent_name)
            
            # Create or update agent
            if agent_name not in self.agents:
                self._create_new_agent(agent_name)
            
            # Extract basic triplet from belief content
            # This is a simplified extraction - in practice, you'd use the full triplet extractor
            if " is " in belief_content:
                parts = belief_content.split(" is ", 1)
                if len(parts) == 2:
                    subject = parts[0].strip()
                    predicate = "is"
                    obj = parts[1].strip()
                else:
                    return None
            elif any(verb in belief_content for verb in [" like", " love", " hate", " prefer"]):
                # Handle preference statements
                for verb in [" like", " love", " hate", " prefer"]:
                    if verb in belief_content:
                        parts = belief_content.split(verb, 1)
                        if len(parts) == 2:
                            subject = parts[0].strip() or agent_name
                            predicate = verb.strip()
                            obj = parts[1].strip()
                            break
                else:
                    return None
            else:
                # Default parsing
                words = belief_content.split()
                if len(words) >= 3:
                    subject = words[0]
                    predicate = words[1] 
                    obj = " ".join(words[2:])
                else:
                    return None
            
            # Create the attributed fact
            fact = EnhancedTripletFact(
                subject=subject,
                predicate=predicate,
                object=obj,
                perspective=agent_name,
                source=speaker,
                confidence=0.7,  # Default confidence for attributed beliefs
                timestamp=time.time()
            )
            
            # Add attribution metadata
            fact.confidence_by_subject[agent_name] = 0.8
            fact.confidence_by_subject[speaker] = 0.6  # Lower confidence from reporter
            
            # Create attribution record
            attribution = BeliefAttribution(
                fact_id=fact.id,
                belief_holder=agent_name,
                attributed_by=speaker,
                confidence=0.7,
                context=original_statement,
                timestamp=time.time()
            )
            
            self.belief_attributions[fact.id].append(attribution)
            
            print(f"[TheoryOfMind] Created attributed fact: {agent_name} believes '{subject} {predicate} {obj}'")
            
            return fact
        
        return None
    
    def _check_for_nested_belief(self, statement: str, speaker: str) -> Optional[EnhancedTripletFact]:
        """Check for nested beliefs like 'I think Anna believes...'."""
        nested_patterns = [
            r"i (?:think|believe) (?:that )?(\w+) (?:thinks|believes) (?:that )?(.+)",
            r"i suspect (?:that )?(\w+) (?:thinks|believes) (?:that )?(.+)",
            r"(\w+) probably (?:thinks|believes) (?:that )?(.+)"
        ]
        
        for pattern in nested_patterns:
            match = re.search(pattern, statement.lower())
            if match:
                agent_name = self._normalize_agent_name(match.group(1))
                belief_content = match.group(2).strip()
                
                # Create nested belief structure
                if agent_name not in self.agents:
                    self._create_new_agent(agent_name)
                
                # Create the core belief
                core_fact = self._parse_belief_content(belief_content, agent_name)
                if core_fact:
                    # Mark as nested belief
                    core_fact.set_nested_belief(speaker, agent_name)
                    
                    # Create nested belief record
                    nested_belief = NestedBelief(
                        belief_id=str(uuid.uuid4()),
                        outer_agent=speaker,
                        inner_agent=agent_name,
                        core_belief=core_fact,
                        nesting_depth=2,
                        confidence_chain=[0.6, 0.7],  # Confidence decreases with nesting
                        original_statement=statement
                    )
                    
                    self.nested_beliefs[nested_belief.belief_id] = nested_belief
                    
                    print(f"[TheoryOfMind] Created nested belief: {speaker} believes {agent_name} believes '{belief_content}'")
                    
                    return core_fact
        
        return None
    
    def _parse_belief_content(self, content: str, agent: str) -> Optional[EnhancedTripletFact]:
        """Parse belief content into a structured fact."""
        # Simplified parsing - replace with full triplet extraction in practice
        content = content.strip()
        
        if " is " in content:
            parts = content.split(" is ", 1)
            subject = parts[0].strip()
            predicate = "is"
            obj = parts[1].strip()
        elif any(verb in content for verb in [" like", " love", " hate", " prefer", " want"]):
            for verb in [" like", " love", " hate", " prefer", " want"]:
                if verb in content:
                    parts = content.split(verb, 1)
                    subject = parts[0].strip() or agent
                    predicate = verb.strip()
                    obj = parts[1].strip()
                    break
            else:
                return None
        else:
            words = content.split()
            if len(words) >= 3:
                subject = words[0]
                predicate = words[1]
                obj = " ".join(words[2:])
            else:
                return None
        
        return EnhancedTripletFact(
            subject=subject,
            predicate=predicate,
            object=obj,
            perspective=agent,
            confidence=0.6
        )
    
    def _normalize_agent_name(self, name: str) -> str:
        """Normalize agent names for consistency."""
        name = name.strip().lower()
        
        # Handle pronouns and references
        pronoun_mapping = {
            "they": "unknown_person",
            "someone": "unknown_person", 
            "everybody": "general_people",
            "everyone": "general_people",
            "people": "general_people"
        }
        
        return pronoun_mapping.get(name, name)
    
    def _create_new_agent(self, agent_name: str):
        """Create a new agent for belief tracking."""
        self.agents[agent_name] = PerspectiveAgent(
            agent_id=agent_name,
            name=agent_name.title(),
            agent_type="person",
            trust_level=0.5,  # Default neutral trust
            consistency_score=0.5,
            last_interaction=time.time()
        )
        
        print(f"[TheoryOfMind] Created new agent: {agent_name}")
    
    def detect_perspective_contradictions(self, facts: List[EnhancedTripletFact]) -> List[Dict]:
        """
        Detect contradictions across different perspectives.
        """
        contradictions = []
        
        # Group facts by subject-object pairs
        fact_groups = defaultdict(list)
        for fact in facts:
            key = (fact.subject.lower(), fact.object.lower())
            fact_groups[key].append(fact)
        
        # Check for contradictions within each group
        for (subject, obj), group_facts in fact_groups.items():
            if len(group_facts) < 2:
                continue
            
            # Look for conflicting predicates from different perspectives
            by_perspective = defaultdict(list)
            for fact in group_facts:
                by_perspective[fact.perspective].append(fact)
            
            # Check across perspectives
            perspectives = list(by_perspective.keys())
            for i, persp1 in enumerate(perspectives):
                for persp2 in perspectives[i+1:]:
                    facts1 = by_perspective[persp1]
                    facts2 = by_perspective[persp2]
                    
                    contradiction = self._check_cross_perspective_contradiction(facts1, facts2)
                    if contradiction:
                        contradictions.append(contradiction)
        
        # Check for deception indicators
        deception_indicators = self._detect_potential_deception(facts)
        contradictions.extend(deception_indicators)
        
        self.perspective_contradictions.extend(contradictions)
        
        return contradictions
    
    def _check_cross_perspective_contradiction(self, facts1: List[EnhancedTripletFact], 
                                            facts2: List[EnhancedTripletFact]) -> Optional[Dict]:
        """Check for contradictions between facts from different perspectives."""
        for fact1 in facts1:
            for fact2 in facts2:
                if self._are_contradictory_predicates(fact1.predicate, fact2.predicate):
                    # Found a cross-perspective contradiction
                    return {
                        'type': 'cross_perspective_contradiction',
                        'fact1': fact1,
                        'fact2': fact2,
                        'perspective1': fact1.perspective,
                        'perspective2': fact2.perspective,
                        'subject': fact1.subject,
                        'object': fact1.object,
                        'conflict': f"{fact1.perspective} says '{fact1.predicate}' but {fact2.perspective} says '{fact2.predicate}'",
                        'severity': self._calculate_contradiction_severity(fact1, fact2),
                        'timestamp': time.time()
                    }
        
        return None
    
    def _detect_potential_deception(self, facts: List[EnhancedTripletFact]) -> List[Dict]:
        """Detect potential deception based on belief patterns."""
        deception_indicators = []
        
        # Look for facts where someone attributes a belief that contradicts known facts
        for fact in facts:
            if fact.source and fact.source != fact.perspective:
                # Someone is reporting what someone else believes
                
                # Check if this conflicts with what we know that person actually believes
                actual_beliefs = [f for f in facts 
                                if f.perspective == fact.perspective and 
                                f.source == fact.perspective and  # Direct from them
                                f.subject == fact.subject and
                                f.object == fact.object]
                
                for actual_belief in actual_beliefs:
                    if self._are_contradictory_predicates(fact.predicate, actual_belief.predicate):
                        deception_indicators.append({
                            'type': 'potential_deception',
                            'reported_fact': fact,
                            'actual_fact': actual_belief,
                            'reporter': fact.source,
                            'subject_agent': fact.perspective,
                            'conflict': f"{fact.source} claims {fact.perspective} believes '{fact.predicate}' but {fact.perspective} actually said '{actual_belief.predicate}'",
                            'deception_probability': 0.7,
                            'timestamp': time.time()
                        })
        
        return deception_indicators
    
    def _are_contradictory_predicates(self, pred1: str, pred2: str) -> bool:
        """Check if two predicates are contradictory."""
        contradictory_pairs = [
            ('like', 'hate'), ('like', 'dislike'),
            ('love', 'hate'), ('love', 'dislike'), 
            ('enjoy', 'hate'), ('enjoy', 'dislike'),
            ('want', 'reject'), ('prefer', 'dislike'),
            ('believe', 'doubt'), ('trust', 'distrust')
        ]
        
        pred1_clean = pred1.lower().strip()
        pred2_clean = pred2.lower().strip()
        
        for pair in contradictory_pairs:
            if (pred1_clean in pair and pred2_clean in pair and pred1_clean != pred2_clean):
                return True
        
        return False
    
    def _calculate_contradiction_severity(self, fact1: EnhancedTripletFact, fact2: EnhancedTripletFact) -> float:
        """Calculate how severe a contradiction is."""
        base_severity = 0.5
        
        # Higher severity if both agents are trusted
        agent1_trust = self.agents.get(fact1.perspective, {}).trust_level or 0.5
        agent2_trust = self.agents.get(fact2.perspective, {}).trust_level or 0.5
        trust_factor = (agent1_trust + agent2_trust) / 2
        
        # Higher severity if facts are recent
        fact1_age = (time.time() - fact1.timestamp) / (24 * 3600) if fact1.timestamp else 30
        fact2_age = (time.time() - fact2.timestamp) / (24 * 3600) if fact2.timestamp else 30
        recency_factor = 1.0 / (1.0 + min(fact1_age, fact2_age) / 7)  # Decay over weeks
        
        # Higher severity if both facts have high confidence
        confidence_factor = (fact1.confidence + fact2.confidence) / 2
        
        severity = base_severity * trust_factor * recency_factor * confidence_factor
        return min(1.0, severity)
    
    def update_agent_trust(self, agent_id: str, trust_adjustment: float, reason: str):
        """Update trust level for an agent based on new information."""
        if agent_id in self.agents:
            old_trust = self.agents[agent_id].trust_level
            new_trust = max(0.0, min(1.0, old_trust + trust_adjustment))
            self.agents[agent_id].trust_level = new_trust
            
            print(f"[TheoryOfMind] Updated trust for {agent_id}: {old_trust:.2f} → {new_trust:.2f} (reason: {reason})")
    
    def get_agent_beliefs(self, agent_id: str) -> List[str]:
        """Get all belief fact IDs attributed to a specific agent."""
        return [attribution.fact_id for fact_list in self.belief_attributions.values() 
                for attribution in fact_list 
                if attribution.belief_holder == agent_id]
    
    def get_perspective_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked perspectives and beliefs."""
        summary = {
            'total_agents': len(self.agents),
            'total_attributions': sum(len(attrs) for attrs in self.belief_attributions.values()),
            'nested_beliefs': len(self.nested_beliefs),
            'perspective_contradictions': len(self.perspective_contradictions),
            'agents': {}
        }
        
        for agent_id, agent in self.agents.items():
            agent_belief_ids = self.get_agent_beliefs(agent_id)
            
            summary['agents'][agent_id] = {
                'name': agent.name,
                'type': agent.agent_type,
                'trust_level': agent.trust_level,
                'consistency_score': agent.consistency_score,
                'belief_count': len(agent_belief_ids),
                'last_interaction': agent.last_interaction,
                'recent_beliefs': [
                    f"Belief ID: {fact_id}"
                    for fact_id in agent_belief_ids[-3:]  # Last 3 beliefs
                ]
            }
        
        return summary
    
    def generate_perspective_insights(self) -> List[str]:
        """Generate insights about perspective patterns and potential issues."""
        insights = []
        
        # Trust level insights
        low_trust_agents = [agent for agent in self.agents.values() if agent.trust_level < 0.3]
        if low_trust_agents:
            insights.append(f"⚠️ {len(low_trust_agents)} agents have low trust levels")
        
        high_trust_agents = [agent for agent in self.agents.values() if agent.trust_level > 0.8]
        if high_trust_agents:
            insights.append(f"✅ {len(high_trust_agents)} agents have high trust levels")
        
        # Contradiction insights
        recent_contradictions = [c for c in self.perspective_contradictions 
                               if time.time() - c['timestamp'] < 24 * 3600]
        if recent_contradictions:
            insights.append(f"🔥 {len(recent_contradictions)} perspective contradictions detected recently")
        
        # Deception insights
        deception_indicators = [c for c in self.perspective_contradictions if c['type'] == 'potential_deception']
        if deception_indicators:
            insights.append(f"🕵️ {len(deception_indicators)} potential deception indicators found")
        
        # Nested belief insights
        if self.nested_beliefs:
            insights.append(f"🧠 Tracking {len(self.nested_beliefs)} nested beliefs")
        
        # Agent activity insights
        active_agents = [agent for agent in self.agents.values() 
                        if agent.last_interaction and time.time() - agent.last_interaction < 7 * 24 * 3600]
        insights.append(f"👥 {len(active_agents)} agents active in the last week")
        
        return insights if insights else ["All perspectives appear stable and consistent."] 