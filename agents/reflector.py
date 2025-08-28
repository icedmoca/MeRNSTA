#!/usr/bin/env python3
"""
ReflectorAgent - Belief summarization and meta-cognitive analysis specialist
"""

import logging
from typing import Dict, Any, List
from .base import BaseAgent

class ReflectorAgent(BaseAgent):
    """
    Specialized agent for summarizing beliefs and providing meta-cognitive analysis.
    
    Capabilities:
    - Belief summarization
    - Pattern identification
    - Meta-cognitive analysis
    - Insight synthesis
    """
    
    def __init__(self):
        super().__init__("reflector")
        self.reflection_depth = self.agent_config.get('depth', 'deep')
        self.focus_patterns = self.agent_config.get('focus_patterns', True)
        self.include_contradictions = self.agent_config.get('include_contradictions', True)
        
    def get_agent_instructions(self) -> str:
        """Return specialized instructions for the reflector agent."""
        return (
            "Your role is to reflect on information, beliefs, and conversations to identify patterns, "
            "synthesize insights, and provide meta-cognitive analysis. Look for themes, contradictions, "
            "and deeper meanings. Help users understand their own thinking patterns and belief systems. "
            f"Reflection depth: {self.reflection_depth}. "
            f"Focus on patterns: {self.focus_patterns}. "
            f"Include contradictions: {self.include_contradictions}."
        )
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """
        Generate reflective analysis of the given content.
        
        Args:
            message: The content to reflect upon
            context: Additional context
            
        Returns:
            Reflective analysis and insights
        """
        if not self.enabled:
            return "[ReflectorAgent] Agent disabled"
            
        try:
            # Get extensive memory context for reflection
            memory_context = self.get_memory_context(message, max_facts=10)
            
            # Check if this is a mathematical topic
            if self.symbolic_engine and self.symbolic_engine.is_symbolic_query(message):
                return self._reflect_on_mathematical_thinking(message)
            
            # Build reflection prompt
            reflection_prompt = self._build_reflection_prompt(message, memory_context, context)
            
            # Generate reflection using LLM
            response = self.generate_llm_response(reflection_prompt)
            
            # Post-process the response
            return self._format_reflection_response(response)
            
        except Exception as e:
            logging.error(f"[ReflectorAgent] Error generating reflection: {e}")
            return f"[ReflectorAgent] Error reflecting on content: {str(e)}"
    
    def _reflect_on_mathematical_thinking(self, topic: str) -> str:
        """Provide reflection on mathematical thinking patterns."""
        try:
            if self.symbolic_engine.is_symbolic_query(topic):
                result = self.symbolic_engine.evaluate(topic)
                return (
                    f"🌟 **Mathematical Reflection**: {topic}\n\n"
                    "**Thinking patterns observed:**\n"
                    "- Logical, step-by-step problem-solving approach\n"
                    "- Reliance on established mathematical rules\n"
                    "- Preference for precise, unambiguous answers\n\n"
                    f"**Computational insight**: The expression evaluates to {result.get('result', 'unknown')}\n"
                    f"**Method used**: {result.get('method', 'unknown')}\n\n"
                    "**Meta-cognitive observation**: Mathematical queries demonstrate systematic thinking "
                    "and trust in computational methods. This suggests comfort with objective, "
                    "rule-based problem solving."
                )
            else:
                return (
                    f"🌟 **Reflection**: {topic}\n\n"
                    "**Pattern observation**: Query appears mathematical but isn't clearly structured.\n"
                    "**Possible meanings**: Could indicate uncertainty about mathematical notation, "
                    "or desire to explore mathematical concepts in natural language.\n\n"
                    "**Insight**: There may be a gap between mathematical intent and expression."
                )
        except Exception as e:
            return f"🌟 **Reflection Error**: Could not analyze mathematical thinking: {e}"
    
    def _build_reflection_prompt(self, message: str, memory_context: list, context: Dict[str, Any] = None) -> str:
        """Build a specialized prompt for reflective analysis."""
        prompt_parts = [
            f"You are the ReflectorAgent. Reflect deeply on this content and identify patterns:",
            f"Content: {message}",
            ""
        ]
        
        if memory_context:
            prompt_parts.extend([
                f"Historical context from memory:",
                f"{'; '.join(memory_context)}",
                ""
            ])
            
        prompt_parts.extend([
            "Provide reflective analysis including:",
            "- Underlying themes and patterns",
            "- Connections to broader concepts",
            "- Meta-cognitive insights"
        ])
        
        if self.focus_patterns:
            prompt_parts.append("- Recurring patterns in thinking or behavior")
            
        if self.include_contradictions:
            prompt_parts.append("- Any contradictions or tensions identified")
            
        prompt_parts.extend([
            "",
            f"Reflection depth: {self.reflection_depth}",
            "Synthesize insights that help understand the deeper meaning."
        ])
        
        if context and context.get('timeframe'):
            prompt_parts.append(f"Focus on patterns over timeframe: {context['timeframe']}")
            
        return "\n".join(prompt_parts)
    
    def _format_reflection_response(self, response: str) -> str:
        """Format the reflection response for consistency."""
        if not response.startswith("🌟"):
            response = f"🌟 **Reflective Analysis**\n\n{response}"
            
        return response
    
    def summarize_belief_system(self, domain: str = None) -> str:
        """Summarize beliefs in a specific domain or overall."""
        memory_context = self.get_memory_context(domain or "beliefs values opinions", max_facts=15)
        
        if not memory_context:
            return "🌟 **Belief Summary**: No significant beliefs identified in memory yet."
            
        prompt = (
            f"Summarize the belief system based on this information:\n"
            f"{'; '.join(memory_context)}\n"
            f"Domain focus: {domain or 'general'}\n"
            "Identify core values, recurring themes, and belief patterns."
        )
        
        response = self.generate_llm_response(prompt)
        return f"🌟 **Belief System Summary**\n\n{response}"
    
    def identify_cognitive_patterns(self, lookback_items: int = 20) -> str:
        """Identify patterns in cognitive behavior and thinking."""
        memory_context = self.get_memory_context("thinking behavior patterns", max_facts=lookback_items)
        
        prompt = (
            f"Analyze cognitive patterns from this information:\n"
            f"{'; '.join(memory_context) if memory_context else 'Limited data available'}\n"
            "Look for patterns in reasoning, decision-making, and problem-solving approaches."
        )
        
        response = self.generate_llm_response(prompt)
        return f"🧠 **Cognitive Pattern Analysis**\n\n{response}"
    
    def synthesize_insights(self, topic: str) -> str:
        """Synthesize higher-level insights about a topic."""
        memory_context = self.get_memory_context(topic, max_facts=10)
        
        prompt = (
            f"Synthesize deeper insights about: {topic}\n"
            f"Based on: {'; '.join(memory_context) if memory_context else 'Current knowledge'}\n"
            "Look beyond surface details to identify fundamental insights and implications."
        )
        
        response = self.generate_llm_response(prompt)
        return f"💡 **Insight Synthesis**\n\n{response}"
    
    def get_reflection_capabilities(self) -> Dict[str, Any]:
        """Get reflection capabilities and current status."""
        return {
            "agent_type": "reflector",
            "reflection_depth": self.reflection_depth,
            "focus_patterns": self.focus_patterns,
            "include_contradictions": self.include_contradictions,
            "enabled": self.enabled,
            "capabilities": [
                "belief_summarization",
                "pattern_identification", 
                "meta_cognitive_analysis",
                "insight_synthesis",
                "drift_triggered_goals"
            ]
        }
    
    def detect_drift_triggered_goals(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Detect token drift events and generate repair meta-goals.
        
        Args:
            limit: Maximum number of recent drift events to analyze
            
        Returns:
            List of drift-triggered goals in dictionary format
        """
        try:
            from agents.cognitive_repair_agent import detect_drift_triggered_goals
            
            # Get drift-triggered goals
            goals = detect_drift_triggered_goals(limit=limit)
            
            # Convert to dictionary format for API compatibility
            goal_dicts = []
            for goal in goals:
                goal_dict = {
                    "type": goal.type,
                    "reason": goal.reason,
                    "token_id": goal.token_id,
                    "cluster_id": goal.cluster_id,
                    "goal": goal.goal,
                    "priority": goal.priority,
                    "timestamp": goal.timestamp,
                    "status": goal.status,
                    "affected_facts": goal.affected_facts,
                    "repair_strategy": goal.repair_strategy,
                    "goal_id": goal.goal_id,
                    "drift_score": goal.drift_score
                }
                goal_dicts.append(goal_dict)
            
            return goal_dicts
            
        except ImportError:
            logging.warning("[ReflectorAgent] CognitiveRepairAgent not available for drift goal detection")
            return []
        except Exception as e:
            logging.error(f"[ReflectorAgent] Error detecting drift-triggered goals: {e}")
            return []
    
    def generate_drift_repair_report(self) -> str:
        """
        Generate a comprehensive report on drift-triggered repair goals.
        
        Returns:
            Formatted report string
        """
        try:
            from agents.cognitive_repair_agent import generate_repair_report
            return generate_repair_report()
        except ImportError:
            return "Cognitive repair agent not available for drift repair reporting."
        except Exception as e:
            return f"Error generating drift repair report: {e}" 