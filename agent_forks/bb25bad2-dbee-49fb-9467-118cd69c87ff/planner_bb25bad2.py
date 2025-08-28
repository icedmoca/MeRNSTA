#!/usr/bin/env python3
"""
PlannerAgent - Task breakdown and step planning specialist
"""

import logging
from typing import Dict, Any
from .base import BaseAgent

class PlannerAgent(BaseAgent):
    """
    Specialized agent for breaking down complex tasks into actionable steps.
    
    Capabilities:
    - Task decomposition
    - Step sequencing
    - Resource identification
    - Timeline estimation
    """
    
    def __init__(self):
        super().__init__("planner")
        self.planning_style = self.agent_config.get('style', 'structured')
        self.max_steps = self.agent_config.get('max_steps', 10)
        
    def get_agent_instructions(self) -> str:
        """Return specialized instructions for the planner agent."""
        return (
            "Your role is to break down complex tasks into clear, actionable steps. "
            "Focus on logical sequence, dependencies, and practical implementation. "
            "Be specific about what needs to be done, in what order, and what resources are needed. "
            f"Planning style: {self.planning_style}. "
            f"Maximum steps: {self.max_steps}."
        )
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """
        Generate a step-by-step plan for the given task.
        
        Args:
            message: The task or goal to plan for
            context: Additional context
            
        Returns:
            Detailed step-by-step plan
        """
        if not self.enabled:
            return "[PlannerAgent] Agent disabled"
            
        try:
            # Get memory context for planning
            memory_context = self.get_memory_context(message, max_facts=3)
            
            # Check if this is a mathematical task
            if self.symbolic_engine and self.symbolic_engine.is_symbolic_query(message):
                return self._plan_mathematical_task(message)
            
            # Build planning prompt
            planning_prompt = self._build_planning_prompt(message, memory_context, context)
            
            # Generate plan using LLM
            response = self.generate_llm_response(planning_prompt)
            
            # Post-process the response to ensure it's properly formatted
            return self._format_plan_response(response)
            
        except Exception as e:
            logging.error(f"[PlannerAgent] Error generating plan: {e}")
            return f"[PlannerAgent] Error creating plan: {str(e)}"
    
    def _plan_mathematical_task(self, task: str) -> str:
        """Create a plan for solving a mathematical problem."""
        return (
            f"📊 Mathematical Problem Analysis: {task}\n\n"
            "**Step-by-step approach:**\n"
            "1. Parse the mathematical expression\n"
            "2. Identify the operations needed\n"
            "3. Apply order of operations (PEMDAS)\n"
            "4. Calculate the result\n"
            "5. Verify the answer\n\n"
            "**Resources needed:** Symbolic computation engine\n"
            "**Estimated time:** < 1 second"
        )
    
    def _build_planning_prompt(self, message: str, memory_context: list, context: Dict[str, Any] = None) -> str:
        """Build a specialized prompt for planning tasks."""
        prompt_parts = [
            f"You are the PlannerAgent. Break down this task into clear, actionable steps:",
            f"Task: {message}",
            "",
            "Provide a structured plan with:",
            "- Clear numbered steps",
            "- Dependencies between steps", 
            "- Required resources",
            "- Estimated timeline",
            "",
            f"Planning style: {self.planning_style}",
            f"Maximum {self.max_steps} steps"
        ]
        
        if memory_context:
            prompt_parts.insert(-3, f"Relevant context: {'; '.join(memory_context)}")
            prompt_parts.insert(-3, "")
            
        if context and context.get('urgency'):
            prompt_parts.append(f"Urgency level: {context['urgency']}")
            
        return "\n".join(prompt_parts)
    
    def _format_plan_response(self, response: str) -> str:
        """Format the planning response for consistency."""
        if not response.startswith("📋"):
            response = f"📋 **Task Planning**\n\n{response}"
            
        return response
    
    def create_milestone_plan(self, goal: str, timeframe: str = None) -> str:
        """Create a milestone-based plan for longer-term goals."""
        prompt = (
            f"Create a milestone-based plan for: {goal}\n"
            f"Timeframe: {timeframe or 'flexible'}\n"
            "Focus on major milestones, key deliverables, and success metrics."
        )
        
        return self.generate_llm_response(prompt)
    
    def get_planning_capabilities(self) -> Dict[str, Any]:
        """Return planning-specific capabilities."""
        base_caps = self.get_capabilities()
        base_caps.update({
            "planning_style": self.planning_style,
            "max_steps": self.max_steps,
            "can_plan_math": self.symbolic_engine is not None,
            "can_use_memory": self.memory_system is not None
        })
        return base_caps 