#!/usr/bin/env python3
"""
CriticAgent - Flaw detection and critical analysis specialist
"""

import logging
from typing import Dict, Any, List
from .base import BaseAgent

class CriticAgent(BaseAgent):
    """
    Specialized agent for finding flaws, weaknesses, and potential issues.
    
    Capabilities:
    - Critical analysis
    - Risk identification
    - Assumption challenging
    - Alternative perspective generation
    """
    
    def __init__(self):
        super().__init__("critic")
        self.criticism_style = self.agent_config.get('style', 'constructive')
        self.focus_areas = self.agent_config.get('focus_areas', ['logic', 'feasibility', 'risks'])
        
    def get_agent_instructions(self) -> str:
        """Return specialized instructions for the critic agent."""
        return (
            "Your role is to identify flaws, weaknesses, and potential issues in ideas, plans, or statements. "
            "Be thorough but constructive in your analysis. Look for logical inconsistencies, "
            "unrealistic assumptions, missing considerations, and potential risks. "
            f"Criticism style: {self.criticism_style}. "
            f"Focus areas: {', '.join(self.focus_areas)}."
        )
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """
        Provide critical analysis of the given input.
        
        Args:
            message: The content to analyze critically
            context: Additional context
            
        Returns:
            Critical analysis with identified issues
        """
        if not self.enabled:
            return "[CriticAgent] Agent disabled"
            
        try:
            # Get memory context for analysis
            memory_context = self.get_memory_context(message, max_facts=3)
            
            # Check if this is a mathematical statement
            if self.symbolic_engine and self.symbolic_engine.is_symbolic_query(message):
                return self._critique_mathematical_statement(message)
            
            # Build criticism prompt
            criticism_prompt = self._build_criticism_prompt(message, memory_context, context)
            
            # Generate critique using LLM
            response = self.generate_llm_response(criticism_prompt)
            
            # Post-process the response
            return self._format_critique_response(response)
            
        except Exception as e:
            logging.error(f"[CriticAgent] Error generating critique: {e}")
            return f"[CriticAgent] Error analyzing content: {str(e)}"
    
    def _critique_mathematical_statement(self, statement: str) -> str:
        """Provide concise critique for math, keeping detailed meta for Thoughts."""
        try:
            if self.symbolic_engine.is_symbolic_query(statement):
                result = self.symbolic_engine.evaluate(statement)
                if result.get('success'):
                    self.last_method = result.get('method')
                    self.last_confidence = result.get('confidence')
                    return f"Result: {result['result']}"
                return "I couldn't compute that."
            return "That doesn't look like a solvable expression."
        except Exception as e:
            return f"I couldn't evaluate that ({e})."
    
    def _build_criticism_prompt(self, message: str, memory_context: list, context: Dict[str, Any] = None) -> str:
        """Build a specialized prompt for critical analysis."""
        prompt_parts = [
            f"You are the CriticAgent. Analyze this content critically and identify potential issues:",
            f"Content: {message}",
            "",
            "Provide critical analysis focusing on:",
        ]
        
        # Add focus areas dynamically
        for area in self.focus_areas:
            prompt_parts.append(f"- {area.title()} issues")
            
        prompt_parts.extend([
            "",
            f"Analysis style: {self.criticism_style}",
            "Be specific about what could go wrong and why."
        ])
        
        if memory_context:
            prompt_parts.insert(-3, f"Relevant context: {'; '.join(memory_context)}")
            prompt_parts.insert(-3, "")
            
        if context and context.get('severity'):
            prompt_parts.append(f"Focus on {context['severity']} severity issues")
            
        return "\n".join(prompt_parts)
    
    def _format_critique_response(self, response: str) -> str:
        """Format the critique response for consistency."""
        if not response.startswith("🔍"):
            response = f"🔍 **Critical Analysis**\n\n{response}"
            
        return response
    
    def analyze_assumptions(self, content: str) -> str:
        """Specifically analyze hidden assumptions in content."""
        prompt = (
            f"Identify hidden assumptions in: {content}\n"
            "Focus on unstated premises, implicit beliefs, and taken-for-granted elements."
        )
        
        response = self.generate_llm_response(prompt)
        return f"🎯 **Assumption Analysis**\n\n{response}"
    
    def identify_risks(self, plan: str) -> str:
        """Identify potential risks in a plan or proposal."""
        prompt = (
            f"Identify potential risks and failure modes in: {plan}\n"
            "Consider both immediate and long-term risks, dependencies, and external factors."
        )
        
        response = self.generate_llm_response(prompt)
        return f"⚠️ **Risk Analysis**\n\n{response}"
    
    def analyze_logical_flaws(self, content: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Analyze content for logical flaws and inconsistencies.
        
        Args:
            content: Content to analyze for logical issues
            context: Additional context for analysis
            
        Returns:
            List of identified logical flaws with descriptions
        """
        context = context or {}
        flaws = []
        
        content_lower = content.lower()
        
        # Check for common logical fallacies
        if any(phrase in content_lower for phrase in ['always', 'never', 'all', 'none', 'everybody', 'nobody']):
            flaws.append({
                'type': 'overgeneralization',
                'description': 'Detected absolute statements that may be overgeneralizations',
                'severity': 'medium',
                'location': 'throughout text'
            })
        
        if 'because' not in content_lower and 'since' not in content_lower and 'therefore' not in content_lower:
            flaws.append({
                'type': 'missing_reasoning',
                'description': 'Lacks clear logical reasoning or causal connections',
                'severity': 'high',
                'location': 'overall structure'
            })
        
        # Check for contradictory statements
        if 'but' in content_lower or 'however' in content_lower:
            # Simple contradiction detection
            sentences = content.split('.')
            if len(sentences) > 1:
                flaws.append({
                    'type': 'potential_contradiction',
                    'description': 'Contains contrasting statements that may contradict',
                    'severity': 'medium',
                    'location': 'sentence structure'
                })
        
        return flaws
    
    def evaluate_contradictions(self, statements: List[str], context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Evaluate a list of statements for contradictions.
        
        Args:
            statements: List of statements to check for contradictions
            context: Additional context
            
        Returns:
            List of identified contradictions
        """
        contradictions = []
        
        for i, stmt1 in enumerate(statements):
            for j, stmt2 in enumerate(statements[i+1:], i+1):
                # Simple contradiction detection
                stmt1_lower = stmt1.lower()
                stmt2_lower = stmt2.lower()
                
                # Check for negation patterns
                if ('not' in stmt1_lower and 'not' not in stmt2_lower) or \
                   ('not' in stmt2_lower and 'not' not in stmt1_lower):
                    # Look for similar keywords
                    words1 = set(stmt1_lower.split())
                    words2 = set(stmt2_lower.split())
                    common_words = words1.intersection(words2)
                    
                    if len(common_words) > 2:  # Some overlap suggests related topics
                        contradictions.append({
                            'type': 'negation_contradiction',
                            'statement1': stmt1,
                            'statement2': stmt2,
                            'description': f'Statements {i+1} and {j+1} appear to contradict via negation',
                            'confidence': 0.6
                        })
        
        return contradictions
    
    def challenge_assumptions(self, content: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Identify and challenge underlying assumptions in content.
        
        Args:
            content: Content to analyze for assumptions
            context: Additional context
            
        Returns:
            List of challenged assumptions
        """
        assumptions = []
        content_lower = content.lower()
        
        # Look for assumption indicators
        assumption_indicators = [
            'obviously', 'clearly', 'of course', 'naturally',
            'it is known that', 'everyone knows', 'it is clear that'
        ]
        
        for indicator in assumption_indicators:
            if indicator in content_lower:
                assumptions.append({
                    'type': 'unstated_assumption',
                    'indicator': indicator,
                    'description': f'Content assumes something is {indicator} without justification',
                    'challenge': f'Why is this {indicator}? What evidence supports this claim?',
                    'severity': 'medium'
                })
        
        # Look for causal assumptions
        causal_words = ['causes', 'results in', 'leads to', 'because of']
        for word in causal_words:
            if word in content_lower:
                assumptions.append({
                    'type': 'causal_assumption',
                    'indicator': word,
                    'description': f'Assumes causal relationship indicated by "{word}"',
                    'challenge': 'Is this causal relationship proven? Could there be other factors?',
                    'severity': 'high'
                })
        
        return assumptions
    
    def log_criticism(self, content: str, criticism: Dict[str, Any], context: Dict[str, Any] = None) -> None:
        """
        Log criticism to memory system for learning and pattern recognition.
        
        Args:
            content: Original content that was criticized
            criticism: The criticism details
            context: Additional context
        """
        try:
            if self.memory_system:
                memory_entry = {
                    'type': 'criticism',
                    'original_content': content[:200] + '...' if len(content) > 200 else content,
                    'criticism_type': criticism.get('type', 'general'),
                    'severity': criticism.get('severity', 'unknown'),
                    'description': criticism.get('description', ''),
                    'timestamp': criticism.get('timestamp', 'unknown'),
                    'context': context or {}
                }
                
                self.memory_system.store_fact(
                    f"criticism_{criticism.get('type', 'general')}",
                    memory_entry
                )
                
                logging.info(f"[{self.name}] Logged criticism of type {criticism.get('type')}")
        except Exception as e:
            logging.warning(f"[{self.name}] Failed to log criticism: {e}")

    def get_criticism_capabilities(self) -> Dict[str, Any]:
        """Return criticism-specific capabilities."""
        base_caps = self.get_capabilities()
        base_caps.update({
            "criticism_style": self.criticism_style,
            "focus_areas": self.focus_areas,
            "can_analyze_math": self.symbolic_engine is not None,
            "specialized_methods": ["analyze_assumptions", "identify_risks", "analyze_logical_flaws", "evaluate_contradictions", "challenge_assumptions"]
        })
        return base_caps 