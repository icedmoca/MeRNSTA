#!/usr/bin/env python3
"""
DebateEngine - Multi-Agent Debate Orchestration for MeRNSTA Phase 21

Orchestrates structured debates between diverse internal agents to surface contradictions,
evaluate competing ideas, and converge on better decisions through dialectical reasoning.
"""

import logging
import uuid
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

from .base import BaseAgent
from config.settings import get_config


class ArgumentType(Enum):
    """Types of arguments in debates."""
    LOGICAL = "logical"
    EMPIRICAL = "empirical"
    ETHICAL = "ethical"
    PRACTICAL = "practical"
    EMOTIONAL = "emotional"
    PRECEDENTIAL = "precedential"


class DebateStance(Enum):
    """Possible stances in a debate."""
    STRONGLY_PRO = "strongly_pro"
    PRO = "pro"
    NEUTRAL = "neutral"
    CON = "con"
    STRONGLY_CON = "strongly_con"


class DebatePhase(Enum):
    """Phases of a debate."""
    INITIALIZATION = "initialization"
    OPENING_ARGUMENTS = "opening_arguments"
    REBUTTAL = "rebuttal"
    CROSS_EXAMINATION = "cross_examination"
    CLOSING_ARGUMENTS = "closing_arguments"
    CONCLUSION = "conclusion"


class ConclusionStrategy(Enum):
    """Strategies for concluding debates."""
    MAJORITY = "majority"
    STRENGTH = "strength"
    CONSENSUS = "consensus"
    SYNTHESIS = "synthesis"


@dataclass
class Argument:
    """Represents a single argument in a debate."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    stance: DebateStance = DebateStance.NEUTRAL
    argument_type: ArgumentType = ArgumentType.LOGICAL
    content: str = ""
    supporting_evidence: List[str] = field(default_factory=list)
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)
    responding_to: Optional[str] = None  # ID of argument this responds to
    strength_score: float = 0.0
    logical_validity: float = 0.0
    novelty_score: float = 0.0
    

@dataclass
class DebateRound:
    """Represents a round of debate."""
    round_number: int
    phase: DebatePhase
    arguments: List[Argument] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    

@dataclass
class DebateResult:
    """Final result of a debate."""
    debate_id: str
    claim: str
    conclusion: str
    winning_stance: Optional[DebateStance]
    confidence: float
    consensus_reached: bool
    key_arguments: List[Argument]
    contradictions_resolved: List[str]
    open_questions: List[str]
    synthesis: Optional[str] = None
    

class DebateEngine(BaseAgent):
    """
    Orchestrates multi-agent debates for claims, goals, and plans.
    
    Capabilities:
    - Structured multi-round debates
    - Argument ranking and evaluation
    - Contradiction surfacing and resolution
    - Multiple conclusion strategies
    - Integration with memory and reflection systems
    """
    
    def __init__(self):
        super().__init__("debate_engine")
        
        # Load debate configuration
        self.config = self._load_debate_config()
        self.enabled = self.config.get('enabled', True)
        self.max_rounds = self.config.get('max_rounds', 3)
        self.debate_agents_count = self.config.get('debate_agents', 5)
        self.default_conclusion_strategy = ConclusionStrategy(
            self.config.get('default_conclusion_strategy', 'strength')
        )
        
        # Scoring weights
        scoring = self.config.get('scoring', {})
        self.logic_weight = scoring.get('logic_weight', 1.0)
        self.evidence_weight = scoring.get('evidence_weight', 0.8)
        self.novelty_weight = scoring.get('novelty_weight', 0.6)
        self.coherence_weight = scoring.get('coherence_weight', 0.9)
        
        # Active debates
        self.active_debates: Dict[str, Dict[str, Any]] = {}
        self.debate_history: List[DebateResult] = []
        
        # Import debate participants (will be initialized on demand)
        self.critic_agent = None
        self.debater_agents: List[Any] = []
        
        logging.info(f"[{self.name}] Initialized debate engine with max {self.max_rounds} rounds")
    
    def get_agent_instructions(self) -> str:
        """Return specialized instructions for the debate engine."""
        return (
            "You are the Debate Engine, orchestrating structured multi-agent debates to surface "
            "contradictions and converge on better decisions. Your role is to facilitate "
            "dialectical reasoning between diverse perspectives, evaluate arguments for logical "
            "validity and strength, and synthesize conclusions from competing viewpoints. "
            "Focus on fair moderation, rigorous evaluation, and helping the system reach "
            "well-reasoned decisions through structured argumentation."
        )
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """Generate debate orchestration responses."""
        context = context or {}
        
        # Build memory context for debate patterns
        memory_context = self.get_memory_context(message)
        
        # Use LLM if available for complex debate questions
        if self.llm_fallback:
            prompt = self.build_agent_prompt(message, memory_context)
            try:
                return self.llm_fallback.process(prompt)
            except Exception as e:
                logging.error(f"[{self.name}] LLM processing failed: {e}")
        
        # Handle debate orchestration requests
        if "debate" in message.lower() and "claim" in context:
            try:
                result = self.initiate_debate(context["claim"], context.get("context", {}))
                return f"Debate initiated for claim: '{context['claim']}'. Result: {result.conclusion} (confidence: {result.confidence:.2f})"
            except Exception as e:
                return f"Debate initiation failed: {str(e)}"
        
        if "conclude" in message.lower() and "debate_id" in context:
            try:
                result = self.conclude_debate(
                    context["debate_id"], 
                    ConclusionStrategy(context.get("strategy", "strength"))
                )
                return f"Debate concluded: {result.conclusion}"
            except Exception as e:
                return f"Debate conclusion failed: {str(e)}"
        
        return "I can orchestrate structured debates on claims, plans, and ideas. Provide a claim to debate or ask about debate management."
    
    def _load_debate_config(self) -> Dict[str, Any]:
        """Load debate engine configuration."""
        try:
            config = get_config()
            return config.get('debate_engine', {
                'enabled': True,
                'max_rounds': 3,
                'debate_agents': 5,
                'default_conclusion_strategy': 'strength',
                'scoring': {
                    'logic_weight': 1.0,
                    'evidence_weight': 0.8,
                    'novelty_weight': 0.6,
                    'coherence_weight': 0.9
                }
            })
        except Exception as e:
            logging.warning(f"[{self.name}] Config loading failed: {e}")
            return {'enabled': True}
    
    def initiate_debate(self, claim: str, context: Dict[str, Any] = None) -> DebateResult:
        """
        Initiate a structured debate on a claim.
        
        Args:
            claim: The claim to debate
            context: Additional context for the debate
            
        Returns:
            Final debate result
        """
        if not self.enabled:
            raise RuntimeError("Debate engine is disabled")
        
        debate_id = str(uuid.uuid4())
        context = context or {}
        
        logging.info(f"[{self.name}] Initiating debate {debate_id} on claim: '{claim}'")
        
        # Initialize debate state
        debate_state = {
            'id': debate_id,
            'claim': claim,
            'context': context,
            'rounds': [],
            'participants': [],
            'started_at': datetime.now(),
            'current_phase': DebatePhase.INITIALIZATION
        }
        
        self.active_debates[debate_id] = debate_state
        
        try:
            # Initialize participants
            self._initialize_debate_participants(debate_id)
            
            # Conduct debate rounds
            for round_num in range(1, self.max_rounds + 1):
                self._conduct_debate_round(debate_id, round_num)
            
            # Conclude debate
            result = self.conclude_debate(debate_id, self.default_conclusion_strategy)
            
            # Store in history
            self.debate_history.append(result)
            
            # Clean up active debate
            if debate_id in self.active_debates:
                del self.active_debates[debate_id]
            
            return result
            
        except Exception as e:
            logging.error(f"[{self.name}] Debate {debate_id} failed: {e}")
            # Clean up failed debate
            if debate_id in self.active_debates:
                del self.active_debates[debate_id]
            raise
    
    def _initialize_debate_participants(self, debate_id: str) -> None:
        """Initialize agents participating in the debate."""
        debate_state = self.active_debates[debate_id]
        claim = debate_state['claim']
        
        # Import and initialize participants
        self._ensure_debate_agents()
        
        # Assign stances to debater agents
        stances = [
            DebateStance.STRONGLY_PRO,
            DebateStance.PRO,
            DebateStance.NEUTRAL,
            DebateStance.CON,
            DebateStance.STRONGLY_CON
        ]
        
        participants = []
        for i, agent in enumerate(self.debater_agents[:self.debate_agents_count]):
            stance = stances[i % len(stances)]
            participants.append({
                'agent': agent,
                'stance': stance,
                'arguments_made': []
            })
        
        # Add critic as participant
        if self.critic_agent:
            participants.append({
                'agent': self.critic_agent,
                'stance': DebateStance.NEUTRAL,
                'arguments_made': []
            })
        
        debate_state['participants'] = participants
        logging.info(f"[{self.name}] Initialized {len(participants)} participants for debate {debate_id}")
    
    def _ensure_debate_agents(self) -> None:
        """Ensure debate agents are available."""
        if not self.critic_agent or not self.debater_agents:
            try:
                # Import agent registry to get agents
                from .registry import get_agent_registry
                registry = get_agent_registry()
                
                self.critic_agent = registry.get_agent('critic')
                
                # Get multiple debater instances or create them
                debater_agent = registry.get_agent('debater')
                if debater_agent:
                    # Create multiple debater instances with different personalities
                    self.debater_agents = [debater_agent]
                    # TODO: Create personality-varied debater instances
                
            except Exception as e:
                logging.warning(f"[{self.name}] Could not initialize debate agents: {e}")
                self.critic_agent = None
                self.debater_agents = []
    
    def _conduct_debate_round(self, debate_id: str, round_num: int) -> None:
        """Conduct a single round of debate."""
        debate_state = self.active_debates[debate_id]
        claim = debate_state['claim']
        
        logging.info(f"[{self.name}] Conducting round {round_num} for debate {debate_id}")
        
        # Determine phase based on round number
        if round_num == 1:
            phase = DebatePhase.OPENING_ARGUMENTS
        elif round_num == self.max_rounds:
            phase = DebatePhase.CLOSING_ARGUMENTS
        else:
            phase = DebatePhase.REBUTTAL
        
        round_obj = DebateRound(round_num, phase)
        
        # Gather arguments from all participants
        for participant in debate_state['participants']:
            try:
                argument = self._generate_argument_from_participant(
                    participant, claim, phase, debate_state
                )
                if argument:
                    round_obj.arguments.append(argument)
                    participant['arguments_made'].append(argument.id)
            except Exception as e:
                logging.warning(f"[{self.name}] Failed to get argument from participant: {e}")
        
        # Score arguments
        self._score_arguments(round_obj.arguments)
        
        debate_state['rounds'].append(round_obj)
        debate_state['current_phase'] = phase
    
    def _generate_argument_from_participant(self, participant: Dict[str, Any], 
                                          claim: str, phase: DebatePhase,
                                          debate_state: Dict[str, Any]) -> Optional[Argument]:
        """Generate an argument from a debate participant."""
        agent = participant['agent']
        stance = participant['stance']
        
        # Build context for argument generation
        context = {
            'claim': claim,
            'stance': stance,
            'phase': phase,
            'previous_arguments': self._get_previous_arguments(debate_state),
            'debate_context': debate_state['context']
        }
        
        try:
            # Use agent's debate capabilities if available
            if hasattr(agent, 'generate_argument'):
                argument_content = agent.generate_argument(claim, stance, context)
            elif hasattr(agent, 'respond'):
                prompt = f"As a {stance.value} participant, argue about: {claim}"
                argument_content = agent.respond(prompt, context)
            else:
                # Fallback: generate basic argument
                argument_content = self._generate_fallback_argument(claim, stance)
            
            if argument_content:
                return Argument(
                    agent_id=agent.name if hasattr(agent, 'name') else str(agent),
                    stance=stance,
                    content=argument_content,
                    confidence=0.7,  # Default confidence
                    argument_type=ArgumentType.LOGICAL
                )
        except Exception as e:
            logging.warning(f"[{self.name}] Failed to generate argument: {e}")
        
        return None
    
    def _get_previous_arguments(self, debate_state: Dict[str, Any]) -> List[Argument]:
        """Get all previous arguments from the debate."""
        all_arguments = []
        for round_obj in debate_state['rounds']:
            all_arguments.extend(round_obj.arguments)
        return all_arguments
    
    def _generate_fallback_argument(self, claim: str, stance: DebateStance) -> str:
        """Generate a fallback argument when agents can't provide one."""
        if stance in [DebateStance.PRO, DebateStance.STRONGLY_PRO]:
            return f"I support the claim '{claim}' based on logical reasoning and available evidence."
        elif stance in [DebateStance.CON, DebateStance.STRONGLY_CON]:
            return f"I oppose the claim '{claim}' due to potential flaws and counterarguments."
        else:
            return f"The claim '{claim}' requires careful consideration of multiple perspectives."
    
    def _score_arguments(self, arguments: List[Argument]) -> None:
        """Score arguments for strength, logic, and novelty."""
        for argument in arguments:
            # Basic scoring (can be enhanced with LLM analysis)
            argument.logical_validity = self._assess_logical_validity(argument)
            argument.strength_score = self._assess_argument_strength(argument)
            argument.novelty_score = self._assess_novelty(argument)
    
    def _assess_logical_validity(self, argument: Argument) -> float:
        """Assess the logical validity of an argument."""
        # Basic heuristics - can be enhanced with logical reasoning
        content = argument.content.lower()
        
        score = 0.5  # Base score
        
        # Positive indicators
        if any(word in content for word in ['because', 'therefore', 'thus', 'since']):
            score += 0.2
        if any(word in content for word in ['evidence', 'data', 'research', 'study']):
            score += 0.2
        if len(argument.supporting_evidence) > 0:
            score += 0.3
        
        # Negative indicators
        if any(word in content for word in ['always', 'never', 'everyone', 'nobody']):
            score -= 0.1
        if '!' in content:
            score -= 0.05  # Emotional language
        
        return max(0.0, min(1.0, score))
    
    def _assess_argument_strength(self, argument: Argument) -> float:
        """Assess the overall strength of an argument."""
        validity = argument.logical_validity
        evidence_strength = min(1.0, len(argument.supporting_evidence) * 0.2)
        confidence = argument.confidence
        
        return (validity * self.logic_weight + 
                evidence_strength * self.evidence_weight + 
                confidence * 0.3) / (self.logic_weight + self.evidence_weight + 0.3)
    
    def _assess_novelty(self, argument: Argument) -> float:
        """Assess the novelty of an argument."""
        # Simple novelty assessment based on content uniqueness
        content_words = set(argument.content.lower().split())
        
        # Compare with previous arguments (simplified)
        novelty_score = 0.8  # Default novelty
        
        # Check for unique concepts or approaches
        if any(word in content_words for word in ['innovative', 'unique', 'novel', 'alternative']):
            novelty_score += 0.2
        
        return min(1.0, novelty_score)
    
    def gather_arguments(self, debate_id: str) -> List[Argument]:
        """Gather all arguments from a debate."""
        if debate_id not in self.active_debates:
            return []
        
        debate_state = self.active_debates[debate_id]
        all_arguments = []
        
        for round_obj in debate_state['rounds']:
            all_arguments.extend(round_obj.arguments)
        
        return all_arguments
    
    def rank_arguments(self, arguments: List[Argument], 
                      by: str = "strength") -> List[Argument]:
        """
        Rank arguments by specified criteria.
        
        Args:
            arguments: List of arguments to rank
            by: Ranking criteria ('logic', 'support', 'novelty', 'strength')
            
        Returns:
            Sorted list of arguments
        """
        if by == "logic":
            return sorted(arguments, key=lambda a: a.logical_validity, reverse=True)
        elif by == "support":
            return sorted(arguments, key=lambda a: len(a.supporting_evidence), reverse=True)
        elif by == "novelty":
            return sorted(arguments, key=lambda a: a.novelty_score, reverse=True)
        else:  # strength
            return sorted(arguments, key=lambda a: a.strength_score, reverse=True)
    
    def conclude_debate(self, debate_id: str, 
                       strategy: ConclusionStrategy = ConclusionStrategy.STRENGTH) -> DebateResult:
        """
        Conclude a debate using specified strategy.
        
        Args:
            debate_id: ID of debate to conclude
            strategy: Strategy for reaching conclusion
            
        Returns:
            Final debate result
        """
        if debate_id not in self.active_debates:
            raise ValueError(f"Debate {debate_id} not found")
        
        debate_state = self.active_debates[debate_id]
        all_arguments = self.gather_arguments(debate_id)
        
        logging.info(f"[{self.name}] Concluding debate {debate_id} with {len(all_arguments)} arguments")
        
        # Apply conclusion strategy
        if strategy == ConclusionStrategy.MAJORITY:
            conclusion_data = self._conclude_by_majority(all_arguments)
        elif strategy == ConclusionStrategy.STRENGTH:
            conclusion_data = self._conclude_by_strength(all_arguments)
        elif strategy == ConclusionStrategy.CONSENSUS:
            conclusion_data = self._conclude_by_consensus(all_arguments)
        else:  # SYNTHESIS
            conclusion_data = self._conclude_by_synthesis(all_arguments)
        
        # Create debate result
        result = DebateResult(
            debate_id=debate_id,
            claim=debate_state['claim'],
            conclusion=conclusion_data['conclusion'],
            winning_stance=conclusion_data.get('winning_stance'),
            confidence=conclusion_data['confidence'],
            consensus_reached=conclusion_data['consensus_reached'],
            key_arguments=self.rank_arguments(all_arguments, "strength")[:3],
            contradictions_resolved=self._identify_resolved_contradictions(all_arguments),
            open_questions=self._identify_open_questions(all_arguments),
            synthesis=conclusion_data.get('synthesis')
        )
        
        logging.info(f"[{self.name}] Debate {debate_id} concluded: {result.conclusion}")
        return result
    
    def _conclude_by_majority(self, arguments: List[Argument]) -> Dict[str, Any]:
        """Conclude debate by majority stance."""
        stance_counts = defaultdict(int)
        for arg in arguments:
            stance_counts[arg.stance] += 1
        
        if not stance_counts:
            return {
                'conclusion': 'No clear conclusion reached',
                'confidence': 0.0,
                'consensus_reached': False
            }
        
        winning_stance = max(stance_counts.keys(), key=lambda s: stance_counts[s])
        total_args = len(arguments)
        confidence = stance_counts[winning_stance] / total_args
        
        return {
            'conclusion': f'Majority supports {winning_stance.value} position',
            'winning_stance': winning_stance,
            'confidence': confidence,
            'consensus_reached': confidence > 0.6
        }
    
    def _conclude_by_strength(self, arguments: List[Argument]) -> Dict[str, Any]:
        """Conclude debate by argument strength."""
        if not arguments:
            return {
                'conclusion': 'No arguments to evaluate',
                'confidence': 0.0,
                'consensus_reached': False
            }
        
        # Group arguments by stance and find strongest
        stance_strengths = defaultdict(list)
        for arg in arguments:
            stance_strengths[arg.stance].append(arg.strength_score)
        
        # Calculate average strength per stance
        stance_avg_strength = {}
        for stance, strengths in stance_strengths.items():
            stance_avg_strength[stance] = sum(strengths) / len(strengths)
        
        winning_stance = max(stance_avg_strength.keys(), 
                           key=lambda s: stance_avg_strength[s])
        confidence = stance_avg_strength[winning_stance]
        
        return {
            'conclusion': f'Strongest arguments support {winning_stance.value} position',
            'winning_stance': winning_stance,
            'confidence': confidence,
            'consensus_reached': confidence > 0.7
        }
    
    def _conclude_by_consensus(self, arguments: List[Argument]) -> Dict[str, Any]:
        """Conclude debate by seeking consensus."""
        # Look for convergence in later arguments
        if len(arguments) < 2:
            return {
                'conclusion': 'Insufficient arguments for consensus analysis',
                'confidence': 0.0,
                'consensus_reached': False
            }
        
        # Analyze if arguments are converging
        mid_point = len(arguments) // 2
        early_stances = [arg.stance for arg in arguments[:mid_point]]
        later_stances = [arg.stance for arg in arguments[mid_point:]]
        
        early_diversity = len(set(early_stances))
        later_diversity = len(set(later_stances))
        
        consensus_reached = later_diversity < early_diversity
        
        if consensus_reached and later_stances:
            # Find most common stance in later arguments
            stance_counts = defaultdict(int)
            for stance in later_stances:
                stance_counts[stance] += 1
            
            consensus_stance = max(stance_counts.keys(), key=lambda s: stance_counts[s])
            confidence = stance_counts[consensus_stance] / len(later_stances)
            
            return {
                'conclusion': f'Consensus emerged around {consensus_stance.value} position',
                'winning_stance': consensus_stance,
                'confidence': confidence,
                'consensus_reached': True
            }
        
        return {
            'conclusion': 'No clear consensus reached despite debate',
            'confidence': 0.3,
            'consensus_reached': False
        }
    
    def _conclude_by_synthesis(self, arguments: List[Argument]) -> Dict[str, Any]:
        """Conclude debate by synthesizing arguments."""
        if not arguments:
            return {
                'conclusion': 'No arguments to synthesize',
                'confidence': 0.0,
                'consensus_reached': False
            }
        
        # Extract key points from all arguments
        pro_points = []
        con_points = []
        neutral_points = []
        
        for arg in arguments:
            if arg.stance in [DebateStance.PRO, DebateStance.STRONGLY_PRO]:
                pro_points.append(arg.content)
            elif arg.stance in [DebateStance.CON, DebateStance.STRONGLY_CON]:
                con_points.append(arg.content)
            else:
                neutral_points.append(arg.content)
        
        # Create synthesis
        synthesis_parts = []
        if pro_points:
            synthesis_parts.append(f"Supporting considerations: {'; '.join(pro_points[:2])}")
        if con_points:
            synthesis_parts.append(f"Opposing considerations: {'; '.join(con_points[:2])}")
        if neutral_points:
            synthesis_parts.append(f"Neutral observations: {'; '.join(neutral_points[:2])}")
        
        synthesis = ". ".join(synthesis_parts)
        
        return {
            'conclusion': 'Debate synthesized multiple perspectives',
            'confidence': 0.6,
            'consensus_reached': True,
            'synthesis': synthesis
        }
    
    def _identify_resolved_contradictions(self, arguments: List[Argument]) -> List[str]:
        """Identify contradictions that were resolved during debate."""
        # Simplified implementation - look for opposing arguments
        contradictions = []
        
        pro_args = [arg for arg in arguments if arg.stance in [DebateStance.PRO, DebateStance.STRONGLY_PRO]]
        con_args = [arg for arg in arguments if arg.stance in [DebateStance.CON, DebateStance.STRONGLY_CON]]
        
        if pro_args and con_args:
            contradictions.append(f"Resolved tension between {len(pro_args)} pro and {len(con_args)} con arguments")
        
        return contradictions
    
    def _identify_open_questions(self, arguments: List[Argument]) -> List[str]:
        """Identify questions that remain open after debate."""
        open_questions = []
        
        # Look for questions in argument content
        for arg in arguments:
            if '?' in arg.content:
                questions = [q.strip() + '?' for q in arg.content.split('?') if q.strip()]
                open_questions.extend(questions[:2])  # Limit to 2 per argument
        
        # Add generic open questions based on debate characteristics
        if len(set(arg.stance for arg in arguments)) > 2:
            open_questions.append("How can multiple valid perspectives be reconciled?")
        
        return open_questions[:5]  # Limit total open questions
    
    def summarize_debate(self, debate_id: str) -> str:
        """
        Summarize a completed debate.
        
        Args:
            debate_id: ID of debate to summarize
            
        Returns:
            Formatted debate summary
        """
        # Find debate in history
        debate_result = None
        for result in self.debate_history:
            if result.debate_id == debate_id:
                debate_result = result
                break
        
        if not debate_result:
            return f"Debate {debate_id} not found in history"
        
        summary_parts = [
            f"🎯 Claim: {debate_result.claim}",
            f"📊 Conclusion: {debate_result.conclusion}",
            f"🎖️ Confidence: {debate_result.confidence:.2f}",
            f"🤝 Consensus: {'Yes' if debate_result.consensus_reached else 'No'}",
            "",
            "🔑 Key Arguments:"
        ]
        
        for i, arg in enumerate(debate_result.key_arguments, 1):
            summary_parts.append(f"  {i}. [{arg.stance.value}] {arg.content[:100]}...")
        
        if debate_result.contradictions_resolved:
            summary_parts.append("\n✅ Contradictions Resolved:")
            for contradiction in debate_result.contradictions_resolved:
                summary_parts.append(f"  • {contradiction}")
        
        if debate_result.open_questions:
            summary_parts.append("\n❓ Open Questions:")
            for question in debate_result.open_questions:
                summary_parts.append(f"  • {question}")
        
        if debate_result.synthesis:
            summary_parts.append(f"\n🔄 Synthesis: {debate_result.synthesis}")
        
        return "\n".join(summary_parts)
    
    def get_debate_statistics(self) -> Dict[str, Any]:
        """Get statistics about debate engine usage."""
        return {
            'total_debates': len(self.debate_history),
            'active_debates': len(self.active_debates),
            'consensus_rate': len([r for r in self.debate_history if r.consensus_reached]) / max(1, len(self.debate_history)),
            'average_confidence': sum(r.confidence for r in self.debate_history) / max(1, len(self.debate_history)),
            'most_common_conclusion_type': self._get_most_common_conclusion_type()
        }
    
    def _get_most_common_conclusion_type(self) -> str:
        """Get the most common type of conclusion reached."""
        if not self.debate_history:
            return "None"
        
        conclusion_types = defaultdict(int)
        for result in self.debate_history:
            if result.winning_stance:
                conclusion_types[result.winning_stance.value] += 1
            else:
                conclusion_types['synthesis'] += 1
        
        if conclusion_types:
            return max(conclusion_types.keys(), key=lambda k: conclusion_types[k])
        return "Unknown"