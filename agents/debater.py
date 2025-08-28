#!/usr/bin/env python3
"""
DebaterAgent - Pros/cons analysis and perspective exploration specialist
"""

import logging
from typing import Dict, Any, List
from .base import BaseAgent

class DebaterAgent(BaseAgent):
    """
    Specialized agent for exploring multiple perspectives and weighing pros/cons.
    
    Capabilities:
    - Multiple perspective analysis
    - Pros and cons identification
    - Argument strength evaluation
    - Dialectical reasoning
    """
    
    def __init__(self, personality_traits: Dict[str, Any] = None):
        super().__init__("debater")
        self.debate_style = self.agent_config.get('style', 'balanced')
        self.perspective_count = self.agent_config.get('perspective_count', 3)
        self.include_counterarguments = self.agent_config.get('include_counterarguments', True)
        
        # Configurable personality traits for debate diversity
        default_traits = {
            'skeptical': 0.5,
            'optimistic': 0.5,
            'emotional': 0.3,
            'logical': 0.8,
            'risk_averse': 0.5,
            'innovative': 0.6,
            'detail_oriented': 0.7,
            'big_picture': 0.6
        }
        self.personality_traits = personality_traits or default_traits
        
        # Stance history for consistency
        self.stance_history = []
        self.argument_patterns = []
        
    def get_agent_instructions(self) -> str:
        """Return specialized instructions for the debater agent."""
        return (
            "Your role is to explore multiple perspectives on topics and provide balanced analysis. "
            "Present pros and cons, different viewpoints, and potential counterarguments. "
            "Be fair to all sides while highlighting the strongest points for each perspective. "
            f"Debate style: {self.debate_style}. "
            f"Target perspectives: {self.perspective_count}. "
            f"Include counterarguments: {self.include_counterarguments}."
        )
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """
        Generate multi-perspective analysis of the given topic.
        
        Args:
            message: The topic or statement to debate
            context: Additional context
            
        Returns:
            Multi-perspective analysis with pros/cons
        """
        if not self.enabled:
            return "[DebaterAgent] Agent disabled"
            
        try:
            # Get memory context for debate
            memory_context = self.get_memory_context(message, max_facts=3)
            
            # Check if this is a mathematical topic
            if self.symbolic_engine and self.symbolic_engine.is_symbolic_query(message):
                return self._debate_mathematical_topic(message)
            
            # Build debate prompt
            debate_prompt = self._build_debate_prompt(message, memory_context, context)
            
            # Generate debate analysis using LLM
            response = self.generate_llm_response(debate_prompt)
            
            # Post-process the response
            return self._format_debate_response(response)
            
        except Exception as e:
            logging.error(f"[DebaterAgent] Error generating debate: {e}")
            return f"[DebaterAgent] Error analyzing perspectives: {str(e)}"
    
    def _debate_mathematical_topic(self, topic: str) -> str:
        """Provide concise outcome for math; keep meta for Thoughts."""
        try:
            if self.symbolic_engine.is_symbolic_query(topic):
                result = self.symbolic_engine.evaluate(topic)
                if result.get('success'):
                    self.last_method = result.get('method')
                    self.last_confidence = result.get('confidence')
                    return f"Answer: {result.get('result')}"
                return "I couldn't compute that."
            return "Clarify the mathematical expression."
        except Exception as e:
            return f"Could not analyze: {e}"
    
    def _build_debate_prompt(self, message: str, memory_context: list, context: Dict[str, Any] = None) -> str:
        """Build a specialized prompt for debate analysis."""
        prompt_parts = [
            f"You are the DebaterAgent. Explore multiple perspectives on this topic:",
            f"Topic: {message}",
            "",
            f"Provide {self.perspective_count} different perspectives with:",
            "- Clear pros and cons for each viewpoint",
            "- Strongest arguments for each side",
            "- Potential areas of agreement"
        ]
        
        if self.include_counterarguments:
            prompt_parts.append("- Counterarguments to each position")
            
        prompt_parts.extend([
            "",
            f"Debate style: {self.debate_style}",
            "Be fair and balanced to all perspectives."
        ])
        
        if memory_context:
            prompt_parts.insert(-3, f"Relevant context: {'; '.join(memory_context)}")
            prompt_parts.insert(-3, "")
            
        if context and context.get('focus'):
            prompt_parts.append(f"Special focus on: {context['focus']}")
            
        return "\n".join(prompt_parts)
    
    def _format_debate_response(self, response: str) -> str:
        """Format the debate response for consistency."""
        if not response.startswith("⚖️"):
            response = f"⚖️ **Multi-Perspective Analysis**\n\n{response}"
            
        return response
    
    def devil_advocate(self, position: str) -> str:
        """Play devil's advocate against a given position."""
        prompt = (
            f"Play devil's advocate against this position: {position}\n"
            "Provide the strongest possible counterarguments, even if you personally agree with the position. "
            "Focus on weaknesses, contradictions, and alternative interpretations."
        )
        
        response = self.generate_llm_response(prompt)
        return f"👿 **Devil's Advocate**\n\n{response}"
    
    def find_common_ground(self, position_a: str, position_b: str) -> str:
        """Find potential areas of agreement between opposing positions."""
        prompt = (
            f"Find common ground between these positions:\n"
            f"Position A: {position_a}\n"
            f"Position B: {position_b}\n"
            "Identify shared values, mutual concerns, and potential areas of compromise."
        )
        
        response = self.generate_llm_response(prompt)
        return f"🤝 **Common Ground Analysis**\n\n{response}"
    
    def evaluate_argument_strength(self, argument: str) -> str:
        """Evaluate the logical strength of an argument."""
        prompt = (
            f"Evaluate the logical strength of this argument: {argument}\n"
            "Consider evidence quality, logical structure, assumptions, and potential weaknesses."
        )
        
        response = self.generate_llm_response(prompt)
        return f"💪 **Argument Strength Analysis**\n\n{response}"
    
    def generate_argument(self, claim: str, stance: Any, context: Dict[str, Any] = None) -> str:
        """
        Generate an argument for a specific claim and stance.
        
        Args:
            claim: The claim to argue about
            stance: The stance to take (from DebateStance enum)
            context: Additional context including previous arguments
            
        Returns:
            Generated argument content
        """
        context = context or {}
        
        # Apply personality traits to argument style
        argument_style = self._determine_argument_style(stance)
        
        # Generate argument based on stance and personality
        stance_str = stance.value if hasattr(stance, 'value') else str(stance)
        
        if 'pro' in stance_str.lower():
            argument = self._generate_pro_argument(claim, stance, argument_style)
        elif 'con' in stance_str.lower():
            argument = self._generate_con_argument(claim, stance, argument_style)
        else:  # NEUTRAL
            argument = self._generate_neutral_argument(claim, argument_style)
        
        return argument
    
    def respond_to_opponent(self, opponent_argument: str, context: Dict[str, Any] = None) -> str:
        """
        Generate a response to an opponent's argument.
        
        Args:
            opponent_argument: The argument to respond to
            context: Additional context including stance and claim
            
        Returns:
            Counter-argument or response
        """
        context = context or {}
        
        # Analyze opponent's argument for weaknesses
        weaknesses = self._identify_argument_weaknesses(opponent_argument)
        
        # Generate response based on personality
        if self.personality_traits.get('logical', 0.8) > 0.7:
            response = self._generate_logical_response(opponent_argument, weaknesses)
        elif self.personality_traits.get('emotional', 0.3) > 0.6:
            response = self._generate_emotional_response(opponent_argument, weaknesses)
        else:
            response = self._generate_balanced_response(opponent_argument, weaknesses)
        
        return response
    
    def _determine_argument_style(self, stance: Any) -> Dict[str, float]:
        """Determine argument style based on personality traits."""
        return {
            'logical_emphasis': self.personality_traits.get('logical', 0.8),
            'emotional_appeal': self.personality_traits.get('emotional', 0.3),
            'detail_focus': self.personality_traits.get('detail_oriented', 0.7),
            'innovation_focus': self.personality_traits.get('innovative', 0.6)
        }
    
    def _generate_pro_argument(self, claim: str, stance: Any, style: Dict[str, float]) -> str:
        """Generate a pro argument for the claim."""
        if style['logical_emphasis'] > 0.7:
            return f"The evidence strongly supports {claim} because logical analysis demonstrates its validity and the framework is sound."
        elif style['emotional_appeal'] > 0.5:
            return f"We must embrace {claim} as it represents important progress and benefits for everyone involved."
        else:
            return f"I support {claim} because it aligns with established principles and offers significant advantages."
    
    def _generate_con_argument(self, claim: str, stance: Any, style: Dict[str, float]) -> str:
        """Generate a con argument against the claim."""
        if style['logical_emphasis'] > 0.7:
            return f"Critical analysis reveals that {claim} is flawed because the logical foundation is questionable and assumptions are unproven."
        elif self.personality_traits.get('risk_averse', 0.5) > 0.6:
            return f"The risks associated with {claim} are significant and potential negative consequences outweigh benefits."
        else:
            return f"I oppose {claim} because it lacks sufficient support and closer inspection reveals significant concerns."
    
    def _generate_neutral_argument(self, claim: str, style: Dict[str, float]) -> str:
        """Generate a neutral argument about the claim."""
        return f"The question of {claim} requires balanced consideration of multiple perspectives, examining both merits and limitations carefully."
    
    def _identify_argument_weaknesses(self, argument: str) -> List[str]:
        """Identify potential weaknesses in an opponent's argument."""
        weaknesses = []
        arg_lower = argument.lower()
        
        if any(word in arg_lower for word in ['always', 'never', 'all', 'none']):
            weaknesses.append('overgeneralization')
        if not any(word in arg_lower for word in ['because', 'evidence', 'research']):
            weaknesses.append('insufficient_evidence')
        if any(word in arg_lower for word in ['must', 'should', 'terrible']):
            weaknesses.append('emotional_language')
        
        return weaknesses
    
    def _generate_logical_response(self, opponent_arg: str, weaknesses: List[str]) -> str:
        """Generate a logical response to opponent's argument."""
        if 'overgeneralization' in weaknesses:
            return "I must challenge this argument because it makes sweeping generalizations without acknowledging exceptions or nuance."
        elif 'insufficient_evidence' in weaknesses:
            return "This position lacks concrete evidence to support its claims. A more rigorous analysis would require supporting data."
        else:
            return "While I understand this perspective, the logical framework requires further examination of underlying assumptions."
    
    def _generate_emotional_response(self, opponent_arg: str, weaknesses: List[str]) -> str:
        """Generate an emotionally-driven response."""
        return "I feel compelled to disagree because this perspective overlooks the real human impact and important considerations for everyone involved."
    
    def _generate_balanced_response(self, opponent_arg: str, weaknesses: List[str]) -> str:
        """Generate a balanced response to opponent's argument."""
        return "While I appreciate this perspective, there may be additional factors to consider. Perhaps we can find common ground through further discussion."

    def get_debate_capabilities(self) -> Dict[str, Any]:
        """Return debate-specific capabilities."""
        base_caps = self.get_capabilities()
        base_caps.update({
            "debate_style": self.debate_style,
            "perspective_count": self.perspective_count,
            "include_counterarguments": self.include_counterarguments,
            "personality_traits": self.personality_traits,
            "specialized_methods": ["devil_advocate", "find_common_ground", "evaluate_argument_strength", "generate_argument", "respond_to_opponent"]
        })
        return base_caps 