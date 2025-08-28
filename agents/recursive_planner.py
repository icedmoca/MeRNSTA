#!/usr/bin/env python3
"""
RecursivePlanner - Advanced multi-level planning agent for MeRNSTA

Provides recursive goal decomposition, intention tracking, and plan optimization.
Enables autonomous, intentional self-improvement and long-term reasoning.
"""

import logging
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from .base import BaseAgent

@dataclass
class PlanStep:
    """Individual step within a plan"""
    step_id: str
    subgoal: str
    why: str
    expected_result: str
    prerequisites: List[str]
    status: str = "pending"  # pending, in_progress, completed, failed
    confidence: float = 0.8
    priority: int = 1
    estimated_duration: Optional[str] = None
    resources_needed: List[str] = None
    # Phase 16 enhancements
    next_step: Optional[str] = None
    fallback_step: Optional[str] = None
    conditional_logic: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    execution_results: List[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.resources_needed is None:
            self.resources_needed = []
        if self.execution_results is None:
            self.execution_results = []

@dataclass
class Plan:
    """Comprehensive plan structure"""
    plan_id: str
    goal_text: str
    steps: List[PlanStep]
    plan_type: str = "sequential"  # sequential, parallel, tree, dag
    created_at: str = None
    updated_at: str = None
    status: str = "draft"  # draft, active, completed, failed, abandoned
    confidence: float = 0.8
    priority: int = 1
    parent_goal_id: Optional[str] = None
    intention_chain: List[str] = None
    success_criteria: List[str] = None
    risk_factors: List[str] = None
    # Phase 16 enhancements
    chained_plans: List[str] = None  # Plans that follow this one
    branching_points: Dict[str, List[str]] = None  # Step ID -> alternative paths
    failure_points: List[Dict[str, Any]] = None  # Historical failure analysis
    progress_checkpoints: List[str] = None  # Step IDs that are checkpoints
    adaptive_triggers: Dict[str, str] = None  # Conditions -> actions
    execution_context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.intention_chain is None:
            self.intention_chain = []
        if self.success_criteria is None:
            self.success_criteria = []
        if self.risk_factors is None:
            self.risk_factors = []
        # Phase 16 enhancements
        if self.chained_plans is None:
            self.chained_plans = []
        if self.branching_points is None:
            self.branching_points = {}
        if self.failure_points is None:
            self.failure_points = []
        if self.progress_checkpoints is None:
            self.progress_checkpoints = []
        if self.adaptive_triggers is None:
            self.adaptive_triggers = {}
        if self.execution_context is None:
            self.execution_context = {}

class RecursivePlanner(BaseAgent):
    """
    Advanced planning agent capable of recursive goal decomposition.
    
    Capabilities:
    - Multi-level goal breakdown
    - Intention chain tracking
    - Plan scoring and optimization
    - Plan reuse and adaptation
    - Integration with memory systems
    """
    
    def __init__(self):
        super().__init__("recursive_planner")
        self.max_recursion_depth = self.agent_config.get('max_recursion_depth', 5)
        self.min_confidence_threshold = self.agent_config.get('min_confidence', 0.6)
        self.plan_similarity_threshold = self.agent_config.get('similarity_threshold', 0.7)
        
        # Initialize storage connections
        self._plan_memory = None
        self._intention_model = None
        
        logging.info(f"[{self.name}] Initialized with depth={self.max_recursion_depth}")
    
    @property
    def plan_memory(self):
        """Lazy-load plan memory storage"""
        if self._plan_memory is None:
            try:
                from storage.plan_memory import PlanMemory
                self._plan_memory = PlanMemory()
            except ImportError as e:
                logging.error(f"[{self.name}] Could not load plan memory: {e}")
        return self._plan_memory
    
    @property
    def intention_model(self):
        """Lazy-load intention model storage"""
        if self._intention_model is None:
            try:
                from storage.intention_model import IntentionModel
                self._intention_model = IntentionModel()
            except ImportError as e:
                logging.error(f"[{self.name}] Could not load intention model: {e}")
        return self._intention_model
    
    def get_agent_instructions(self) -> str:
        """Return specialized instructions for the recursive planner agent."""
        return (
            "You are a recursive planning specialist. Your role is to decompose complex goals "
            "into manageable, actionable steps with clear intention chains. Focus on creating "
            "logical hierarchies, identifying dependencies, and tracking the 'why' behind each step. "
            "Consider reusing successful plans and learning from past experiences. "
            f"Maximum recursion depth: {self.max_recursion_depth}. "
            f"Minimum confidence threshold: {self.min_confidence_threshold}."
        )
    
    def plan_goal(self, goal_text: str, parent_goal_id: Optional[str] = None, 
                  context: Optional[Dict[str, Any]] = None, depth: int = 0) -> Plan:
        """
        Generate a multi-step plan for achieving the given goal.
        
        Args:
            goal_text: The goal to plan for
            parent_goal_id: ID of parent goal if this is a subgoal
            context: Additional context for planning
            depth: Current recursion depth
            
        Returns:
            Comprehensive Plan object with steps and metadata
        """
        if not self.enabled:
            raise ValueError(f"[{self.name}] Agent disabled")
        
        if depth >= self.max_recursion_depth:
            logging.warning(f"[{self.name}] Max recursion depth reached for goal: {goal_text}")
            return self._create_simple_plan(goal_text, parent_goal_id)
        
        try:
            # Check for similar existing plans first
            similar_plan = self.reuse_similar_plan(goal_text)
            if similar_plan:
                logging.info(f"[{self.name}] Reusing similar plan for: {goal_text}")
                return self._adapt_existing_plan(similar_plan, goal_text, parent_goal_id)
            
            # Generate new plan using LLM
            plan_prompt = self._build_planning_prompt(goal_text, context, depth)
            response = self._generate_plan_with_llm(plan_prompt)
            
            # Parse response into Plan structure
            plan = self._parse_plan_response(response, goal_text, parent_goal_id)
            
            # Add intention tracking
            if self.intention_model and parent_goal_id:
                self.intention_model.link_goals(plan.plan_id, parent_goal_id, 
                                              f"Subgoal for achieving: {goal_text}")
            
            # Store plan in memory
            if self.plan_memory:
                self.plan_memory.store_plan(plan)
            
            # Recursively plan complex substeps if needed
            self._recursive_subplan_if_needed(plan, depth + 1)
            
            return plan
            
        except Exception as e:
            logging.error(f"[{self.name}] Error planning goal '{goal_text}': {e}")
            return self._create_fallback_plan(goal_text, parent_goal_id)
    
    def score_plan(self, plan: Plan) -> float:
        """
        Score a plan based on feasibility, clarity, and potential success.
        
        Args:
            plan: Plan to evaluate
            
        Returns:
            Score between 0.0 and 1.0
        """
        if not plan or not plan.steps:
            return 0.0
        
        try:
            scores = []
            
            # Step quality scoring (40% weight)
            step_scores = []
            for step in plan.steps:
                step_score = self._score_individual_step(step)
                step_scores.append(step_score)
            
            if step_scores:
                scores.append(sum(step_scores) / len(step_scores) * 0.4)
            
            # Plan coherence scoring (30% weight)
            coherence_score = self._score_plan_coherence(plan)
            scores.append(coherence_score * 0.3)
            
            # Feasibility scoring (20% weight)
            feasibility_score = self._score_plan_feasibility(plan)
            scores.append(feasibility_score * 0.2)
            
            # Intention clarity scoring (10% weight)
            intention_score = self._score_intention_clarity(plan)
            scores.append(intention_score * 0.1)
            
            final_score = sum(scores)
            logging.info(f"[{self.name}] Plan scored: {final_score:.3f} for '{plan.goal_text}'")
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logging.error(f"[{self.name}] Error scoring plan: {e}")
            return 0.5  # Neutral score on error
    
    def execute_plan(self, plan: Plan) -> Dict[str, Any]:
        """
        Execute a plan step by step, tracking progress and outcomes.
        
        Args:
            plan: Plan to execute
            
        Returns:
            Execution results with status, outcomes, and metrics
        """
        if not plan or not plan.steps:
            return {"success": False, "error": "Invalid or empty plan"}
        
        try:
            execution_start = datetime.now()
            plan.status = "active"
            plan.updated_at = execution_start.isoformat()
            
            results = {
                "plan_id": plan.plan_id,
                "goal_text": plan.goal_text,
                "execution_start": execution_start.isoformat(),
                "steps_executed": [],
                "steps_failed": [],
                "overall_success": False,
                "completion_percentage": 0.0,
                "execution_log": []
            }
            
            completed_steps = 0
            total_steps = len(plan.steps)
            
            for i, step in enumerate(plan.steps):
                self._log_execution(results, f"Starting step {i+1}/{total_steps}: {step.subgoal}")
                
                # Check prerequisites
                if not self._check_prerequisites(step, plan.steps[:i]):
                    self._log_execution(results, f"Prerequisites not met for step: {step.subgoal}")
                    step.status = "failed"
                    results["steps_failed"].append({
                        "step_id": step.step_id,
                        "subgoal": step.subgoal,
                        "reason": "Prerequisites not met"
                    })
                    continue
                
                # Execute step
                step.status = "in_progress"
                step_result = self._execute_individual_step(step, plan)
                
                if step_result.get("success", False):
                    step.status = "completed"
                    completed_steps += 1
                    results["steps_executed"].append({
                        "step_id": step.step_id,
                        "subgoal": step.subgoal,
                        "result": step_result.get("result", "Completed")
                    })
                    self._log_execution(results, f"Completed step: {step.subgoal}")
                else:
                    step.status = "failed"
                    results["steps_failed"].append({
                        "step_id": step.step_id,
                        "subgoal": step.subgoal,
                        "reason": step_result.get("error", "Execution failed")
                    })
                    self._log_execution(results, f"Failed step: {step.subgoal} - {step_result.get('error', 'Unknown error')}")
            
            # Calculate final results
            results["completion_percentage"] = (completed_steps / total_steps) * 100
            results["overall_success"] = completed_steps == total_steps
            results["execution_end"] = datetime.now().isoformat()
            
            # Update plan status
            if results["overall_success"]:
                plan.status = "completed"
            elif completed_steps > 0:
                plan.status = "partially_completed"
            else:
                plan.status = "failed"
            
            plan.updated_at = datetime.now().isoformat()
            
            # Record outcome in plan memory
            if self.plan_memory:
                self.plan_memory.record_plan_outcome(plan.plan_id, results)
            
            self._log_execution(results, f"Plan execution completed. Success: {results['overall_success']}")
            
            return results
            
        except Exception as e:
            logging.error(f"[{self.name}] Error executing plan: {e}")
            return {
                "success": False,
                "error": str(e),
                "plan_id": plan.plan_id if plan else None
            }
    
    def reuse_similar_plan(self, goal_text: str) -> Optional[Plan]:
        """
        Find and return a similar plan that can be reused or adapted.
        
        Args:
            goal_text: Goal to find similar plans for
            
        Returns:
            Similar plan if found, None otherwise
        """
        try:
            if not self.plan_memory:
                return None
            
            similar_plans = self.plan_memory.get_similar_plans(goal_text)
            
            if not similar_plans:
                return None
            
            # Filter by success rate and similarity
            viable_plans = [
                plan for plan in similar_plans 
                if plan.get('similarity_score', 0) >= self.plan_similarity_threshold
                and plan.get('success_rate', 0) > 0.5
            ]
            
            if not viable_plans:
                return None
            
            # Return the most successful and similar plan
            best_plan = max(viable_plans, 
                          key=lambda p: (p.get('success_rate', 0) * 0.7 + 
                                       p.get('similarity_score', 0) * 0.3))
            
            logging.info(f"[{self.name}] Found reusable plan with {best_plan.get('similarity_score', 0):.2f} similarity")
            
            return self.plan_memory.get_plan_by_id(best_plan['plan_id'])
            
        except Exception as e:
            logging.error(f"[{self.name}] Error finding similar plans: {e}")
            return None
    
    def _build_planning_prompt(self, goal_text: str, context: Optional[Dict], depth: int) -> str:
        """Build a comprehensive prompt for plan generation"""
        prompt_parts = [
            f"Create a detailed, actionable plan to achieve this goal: {goal_text}",
            "",
            "For each step, provide:",
            "- subgoal: What specifically needs to be done",
            "- why: The reasoning behind this step",
            "- expected_result: What outcome this step should produce",
            "- prerequisites: What must be completed before this step (if any)",
            "- resources_needed: Tools, skills, or materials required",
            "",
            "Consider:",
            "- Break complex tasks into manageable steps",
            "- Identify dependencies between steps",
            "- Be specific about expected outcomes",
            "- Include quality checks and verification steps",
            "- Think about potential risks and mitigation",
        ]
        
        if context:
            prompt_parts.extend([
                "",
                "Additional context:",
                json.dumps(context, indent=2)
            ])
        
        if depth > 0:
            prompt_parts.extend([
                "",
                f"This is a subgoal at depth {depth}. Keep steps focused and specific."
            ])
        
        prompt_parts.extend([
            "",
            "Respond with a JSON structure containing:",
            "- goal_text: The goal being planned",
            "- plan_type: 'sequential', 'parallel', or 'tree'",
            "- steps: Array of step objects",
            "- success_criteria: How to measure successful completion",
            "- risk_factors: Potential challenges or obstacles"
        ])
        
        return "\n".join(prompt_parts)
    
    def _generate_plan_with_llm(self, prompt: str) -> str:
        """Generate plan using LLM"""
        try:
            if self.llm_fallback:
                return self.llm_fallback.generate_response(prompt)
            else:
                # Fallback to a basic structured response
                return self._generate_fallback_response(prompt)
        except Exception as e:
            logging.error(f"[{self.name}] LLM generation failed: {e}")
            return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a basic structured response when LLM is unavailable"""
        # Extract goal from prompt
        lines = prompt.split('\n')
        goal_line = next((line for line in lines if 'goal:' in line.lower()), "")
        goal = goal_line.split(':', 1)[1].strip() if ':' in goal_line else "Complete task"
        
        fallback_plan = {
            "goal_text": goal,
            "plan_type": "sequential",
            "steps": [
                {
                    "subgoal": f"Analyze requirements for: {goal}",
                    "why": "Understanding the full scope is essential for success",
                    "expected_result": "Clear understanding of requirements and constraints",
                    "prerequisites": [],
                    "resources_needed": ["analysis tools", "domain knowledge"]
                },
                {
                    "subgoal": f"Execute core actions for: {goal}",
                    "why": "Implementation of the primary goal activities",
                    "expected_result": "Significant progress toward goal completion",
                    "prerequisites": ["Analyze requirements"],
                    "resources_needed": ["appropriate tools", "sufficient time"]
                },
                {
                    "subgoal": f"Verify completion of: {goal}",
                    "why": "Ensure quality and completeness of work",
                    "expected_result": "Confirmed successful goal achievement",
                    "prerequisites": ["Execute core actions"],
                    "resources_needed": ["verification methods"]
                }
            ],
            "success_criteria": [f"Goal '{goal}' is fully achieved"],
            "risk_factors": ["Insufficient resources", "Unclear requirements", "Time constraints"]
        }
        
        return json.dumps(fallback_plan, indent=2)
    
    def _parse_plan_response(self, response: str, goal_text: str, parent_goal_id: Optional[str]) -> Plan:
        """Parse LLM response into Plan structure"""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = response[json_start:json_end]
                plan_data = json.loads(json_content)
            else:
                # Fallback parsing
                plan_data = {"goal_text": goal_text, "steps": [], "plan_type": "sequential"}
            
            # Create plan steps
            steps = []
            for i, step_data in enumerate(plan_data.get('steps', [])):
                step = PlanStep(
                    step_id=str(uuid.uuid4()),
                    subgoal=step_data.get('subgoal', f'Step {i+1}'),
                    why=step_data.get('why', 'Necessary for goal completion'),
                    expected_result=step_data.get('expected_result', 'Progress toward goal'),
                    prerequisites=step_data.get('prerequisites', []),
                    resources_needed=step_data.get('resources_needed', [])
                )
                steps.append(step)
            
            # Create the plan
            plan = Plan(
                plan_id=str(uuid.uuid4()),
                goal_text=goal_text,
                steps=steps,
                plan_type=plan_data.get('plan_type', 'sequential'),
                parent_goal_id=parent_goal_id,
                success_criteria=plan_data.get('success_criteria', []),
                risk_factors=plan_data.get('risk_factors', [])
            )
            
            return plan
            
        except Exception as e:
            logging.error(f"[{self.name}] Error parsing plan response: {e}")
            return self._create_fallback_plan(goal_text, parent_goal_id)
    
    def _create_simple_plan(self, goal_text: str, parent_goal_id: Optional[str]) -> Plan:
        """Create a simple single-step plan"""
        step = PlanStep(
            step_id=str(uuid.uuid4()),
            subgoal=goal_text,
            why="Direct goal achievement",
            expected_result="Goal completed",
            prerequisites=[]
        )
        
        return Plan(
            plan_id=str(uuid.uuid4()),
            goal_text=goal_text,
            steps=[step],
            parent_goal_id=parent_goal_id
        )
    
    def _create_fallback_plan(self, goal_text: str, parent_goal_id: Optional[str]) -> Plan:
        """Create a basic fallback plan when generation fails"""
        steps = [
            PlanStep(
                step_id=str(uuid.uuid4()),
                subgoal=f"Analyze: {goal_text}",
                why="Understanding is prerequisite to action",
                expected_result="Clear action path identified",
                prerequisites=[]
            ),
            PlanStep(
                step_id=str(uuid.uuid4()),
                subgoal=f"Execute: {goal_text}",
                why="Direct action toward goal",
                expected_result="Goal achieved",
                prerequisites=["Analyze"]
            )
        ]
        
        return Plan(
            plan_id=str(uuid.uuid4()),
            goal_text=goal_text,
            steps=steps,
            parent_goal_id=parent_goal_id
        )
    
    def _adapt_existing_plan(self, existing_plan: Plan, new_goal: str, parent_goal_id: Optional[str]) -> Plan:
        """Adapt an existing plan for a new but similar goal"""
        adapted_steps = []
        for step in existing_plan.steps:
            adapted_step = PlanStep(
                step_id=str(uuid.uuid4()),
                subgoal=step.subgoal.replace(existing_plan.goal_text, new_goal),
                why=step.why,
                expected_result=step.expected_result,
                prerequisites=step.prerequisites,
                resources_needed=step.resources_needed.copy() if step.resources_needed else []
            )
            adapted_steps.append(adapted_step)
        
        return Plan(
            plan_id=str(uuid.uuid4()),
            goal_text=new_goal,
            steps=adapted_steps,
            plan_type=existing_plan.plan_type,
            parent_goal_id=parent_goal_id,
            success_criteria=existing_plan.success_criteria.copy() if existing_plan.success_criteria else [],
            risk_factors=existing_plan.risk_factors.copy() if existing_plan.risk_factors else []
        )
    
    def _recursive_subplan_if_needed(self, plan: Plan, depth: int):
        """Check if any steps need recursive sub-planning"""
        for step in plan.steps:
            # Heuristic: if a step is complex (long description, multiple resources), sub-plan it
            complexity_score = len(step.subgoal.split()) + len(step.resources_needed or [])
            
            if complexity_score > 10 and depth < self.max_recursion_depth:
                try:
                    subplan = self.plan_goal(step.subgoal, plan.plan_id, depth=depth)
                    # Store reference to subplan in step
                    if not hasattr(step, 'subplan_id'):
                        step.subplan_id = subplan.plan_id
                except Exception as e:
                    logging.warning(f"[{self.name}] Failed to create subplan for step: {e}")
    
    def _score_individual_step(self, step: PlanStep) -> float:
        """Score an individual step for quality"""
        score = 0.5  # Base score
        
        # Clear subgoal (+0.2)
        if step.subgoal and len(step.subgoal.strip()) > 5:
            score += 0.2
        
        # Good reasoning (+0.1)
        if step.why and len(step.why.strip()) > 10:
            score += 0.1
        
        # Clear expected result (+0.1)
        if step.expected_result and len(step.expected_result.strip()) > 5:
            score += 0.1
        
        # Prerequisites considered (+0.05)
        if isinstance(step.prerequisites, list):
            score += 0.05
        
        # Resources identified (+0.05)
        if step.resources_needed and len(step.resources_needed) > 0:
            score += 0.05
        
        return min(1.0, score)
    
    def _score_plan_coherence(self, plan: Plan) -> float:
        """Score plan for logical flow and coherence"""
        if not plan.steps:
            return 0.0
        
        coherence_score = 0.5
        
        # Sequential logic check
        if plan.plan_type == "sequential":
            coherence_score += 0.3
        
        # Prerequisites alignment
        prereq_alignment = self._check_prerequisite_alignment(plan.steps)
        coherence_score += prereq_alignment * 0.2
        
        return min(1.0, coherence_score)
    
    def _score_plan_feasibility(self, plan: Plan) -> float:
        """Score plan for feasibility"""
        feasibility_score = 0.7  # Optimistic base
        
        # Check for unrealistic steps
        for step in plan.steps:
            if any(keyword in step.subgoal.lower() for keyword in ['impossible', 'cannot', 'never']):
                feasibility_score -= 0.2
        
        return max(0.0, min(1.0, feasibility_score))
    
    def _score_intention_clarity(self, plan: Plan) -> float:
        """Score plan for clear intention chain"""
        if not plan.steps:
            return 0.0
        
        clear_intentions = sum(1 for step in plan.steps if step.why and len(step.why.strip()) > 5)
        return clear_intentions / len(plan.steps)
    
    def _check_prerequisite_alignment(self, steps: List[PlanStep]) -> float:
        """Check how well prerequisites align with step order"""
        if len(steps) <= 1:
            return 1.0
        
        aligned = 0
        total_checks = 0
        
        for i, step in enumerate(steps):
            if step.prerequisites:
                for prereq in step.prerequisites:
                    total_checks += 1
                    # Check if prerequisite appears in earlier steps
                    for j in range(i):
                        if prereq.lower() in steps[j].subgoal.lower():
                            aligned += 1
                            break
        
        return (aligned / total_checks) if total_checks > 0 else 1.0
    
    def _check_prerequisites(self, step: PlanStep, completed_steps: List[PlanStep]) -> bool:
        """Check if step prerequisites are satisfied"""
        if not step.prerequisites:
            return True
        
        completed_subgoals = [s.subgoal.lower() for s in completed_steps if s.status == "completed"]
        
        for prereq in step.prerequisites:
            prereq_satisfied = any(prereq.lower() in subgoal for subgoal in completed_subgoals)
            if not prereq_satisfied:
                return False
        
        return True
    
    def _execute_individual_step(self, step: PlanStep, plan: Plan) -> Dict[str, Any]:
        """Execute an individual step (placeholder for actual execution)"""
        # This is a placeholder - in a real system, this would:
        # 1. Determine the appropriate execution method
        # 2. Call relevant agents or tools
        # 3. Monitor execution progress
        # 4. Return actual results
        
        try:
            # Simulate execution based on step complexity
            complexity = len(step.subgoal.split()) + len(step.resources_needed or [])
            success_probability = max(0.3, 1.0 - (complexity * 0.05))
            
            # For now, simulate success based on step quality
            import random
            random.seed(hash(step.step_id))  # Deterministic for testing
            
            if random.random() < success_probability:
                return {
                    "success": True,
                    "result": f"Successfully completed: {step.subgoal}",
                    "execution_time": complexity * 0.1  # Simulated time
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to complete: {step.subgoal}",
                    "reason": "Simulated execution failure"
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _log_execution(self, results: Dict, message: str):
        """Add execution log entry"""
        results["execution_log"].append({
            "timestamp": datetime.now().isoformat(),
            "message": message
        })
        logging.info(f"[{self.name}] {message}")
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """Handle direct messages to the planner"""
        if not self.enabled:
            return f"[{self.name}] Agent disabled"
        
        try:
            # Parse planning requests
            if message.lower().startswith(('plan:', 'create plan:', 'generate plan:')):
                goal = message.split(':', 1)[1].strip()
                plan = self.plan_goal(goal, context=context)
                return f"Plan created for '{goal}': {len(plan.steps)} steps, confidence: {plan.confidence:.2f}"
            
            elif message.lower().startswith(('score plan:', 'evaluate plan:')):
                # This would need plan retrieval logic
                return "Plan scoring requires plan ID or plan object"
            
            else:
                # General planning assistance
                return self._provide_planning_guidance(message, context)
        
        except Exception as e:
            logging.error(f"[{self.name}] Error in respond: {e}")
            return f"[{self.name}] Error: {str(e)}"
    
    def _provide_planning_guidance(self, message: str, context: Optional[Dict]) -> str:
        """Provide general planning guidance"""
        guidance = [
            f"I'm the {self.name} agent. I can help you with:",
            "• Breaking down complex goals into actionable steps",
            "• Creating detailed plans with intention tracking",
            "• Scoring and optimizing plans",
            "• Reusing successful plans for similar goals",
            "",
            "Try: 'plan: <your goal>' to get started"
        ]
        
        return "\n".join(guidance)
    
    # === Phase 16 Enhancements: Chaining, Progress Tracking, and Retry Logic ===
    
    def chain_plans(self, current_plan: Plan, next_plan_goals: List[str]) -> None:
        """
        Chain multiple plans together for complex multi-phase goals.
        
        Args:
            current_plan: The current plan to extend
            next_plan_goals: Goals for subsequent plans
        """
        logging.info(f"[{self.name}] Chaining plans from {current_plan.plan_id}")
        
        for goal in next_plan_goals:
            next_plan = self.plan_goal(goal, parent_goal_id=current_plan.plan_id)
            current_plan.chained_plans.append(next_plan.plan_id)
            
            # Set up conditional triggers
            last_step = current_plan.steps[-1] if current_plan.steps else None
            if last_step:
                last_step.next_step = next_plan.steps[0].step_id if next_plan.steps else None
        
        current_plan.updated_at = datetime.now().isoformat()
        
        if self.plan_memory:
            self.plan_memory.store_plan(current_plan)
    
    def add_branching_logic(self, plan: Plan, step_id: str, conditions: Dict[str, str], 
                           alternative_paths: Dict[str, List[str]]) -> None:
        """
        Add branching logic to a plan step.
        
        Args:
            plan: Plan to modify
            step_id: Step that has branching
            conditions: Condition checks
            alternative_paths: Alternative step sequences for each condition
        """
        step = self._find_step_by_id(plan, step_id)
        if not step:
            logging.warning(f"[{self.name}] Step {step_id} not found for branching")
            return
        
        # Store branching information
        plan.branching_points[step_id] = list(alternative_paths.keys())
        
        # Add conditional logic to the step
        step.conditional_logic = json.dumps(conditions)
        
        # Create alternative steps
        for condition, path_steps in alternative_paths.items():
            for i, alt_step_goal in enumerate(path_steps):
                alt_step_id = f"{step_id}_alt_{condition}_{i}"
                alt_step = PlanStep(
                    step_id=alt_step_id,
                    subgoal=alt_step_goal,
                    why=f"Alternative path for condition: {condition}",
                    expected_result=f"Outcome of {alt_step_goal}",
                    prerequisites=[step_id],
                    confidence=step.confidence * 0.9  # Slightly lower confidence for alternatives
                )
                plan.steps.append(alt_step)
        
        plan.updated_at = datetime.now().isoformat()
        
        if self.plan_memory:
            self.plan_memory.store_plan(plan)
    
    def track_plan_progress(self, plan: Plan) -> Dict[str, Any]:
        """
        Track and analyze plan execution progress.
        
        Args:
            plan: Plan to analyze
            
        Returns:
            Progress analysis and recommendations
        """
        if not plan.steps:
            return {"progress": 0.0, "status": "empty"}
        
        completed_steps = [s for s in plan.steps if s.status == "completed"]
        failed_steps = [s for s in plan.steps if s.status == "failed"]
        in_progress_steps = [s for s in plan.steps if s.status == "in_progress"]
        pending_steps = [s for s in plan.steps if s.status == "pending"]
        
        progress_percentage = len(completed_steps) / len(plan.steps) * 100
        
        # Identify bottlenecks
        bottlenecks = []
        for step in in_progress_steps:
            if step.started_at:
                duration = (datetime.now() - datetime.fromisoformat(step.started_at.replace('Z', '+00:00'))).total_seconds() / 60
                if duration > 60:  # More than 1 hour
                    bottlenecks.append({
                        "step_id": step.step_id,
                        "duration_minutes": duration,
                        "goal": step.subgoal
                    })
        
        # Check for blocked steps
        blocked_steps = []
        for step in pending_steps:
            deps_satisfied = all(
                any(s.step_id == dep and s.status == "completed" for s in plan.steps)
                for dep in step.prerequisites
            )
            if not deps_satisfied:
                blocked_steps.append({
                    "step_id": step.step_id,
                    "missing_deps": [dep for dep in step.prerequisites 
                                   if not any(s.step_id == dep and s.status == "completed" for s in plan.steps)]
                })
        
        # Calculate estimated completion
        remaining_effort = sum(
            int(step.estimated_duration or "30") 
            for step in pending_steps + in_progress_steps
        )
        
        progress_analysis = {
            "plan_id": plan.plan_id,
            "progress_percentage": progress_percentage,
            "status": self._determine_plan_status(plan),
            "completed_steps": len(completed_steps),
            "failed_steps": len(failed_steps),
            "in_progress_steps": len(in_progress_steps),
            "pending_steps": len(pending_steps),
            "bottlenecks": bottlenecks,
            "blocked_steps": blocked_steps,
            "estimated_completion_minutes": remaining_effort,
            "failure_points": plan.failure_points,
            "checkpoints_reached": [
                cp for cp in plan.progress_checkpoints 
                if any(s.step_id == cp and s.status == "completed" for s in plan.steps)
            ]
        }
        
        # Generate recommendations
        recommendations = self._generate_progress_recommendations(progress_analysis)
        progress_analysis["recommendations"] = recommendations
        
        return progress_analysis
    
    def handle_step_failure(self, plan: Plan, failed_step_id: str, 
                           failure_reason: str, retry: bool = True) -> Dict[str, Any]:
        """
        Handle step failure with retry logic and fallback options.
        
        Args:
            plan: Plan containing the failed step
            failed_step_id: ID of the step that failed
            failure_reason: Reason for failure
            retry: Whether to attempt retry
            
        Returns:
            Recovery action and status
        """
        step = self._find_step_by_id(plan, failed_step_id)
        if not step:
            return {"error": "Step not found"}
        
        logging.warning(f"[{self.name}] Step {failed_step_id} failed: {failure_reason}")
        
        # Record failure
        failure_record = {
            "step_id": failed_step_id,
            "reason": failure_reason,
            "timestamp": datetime.now().isoformat(),
            "retry_count": step.retry_count
        }
        plan.failure_points.append(failure_record)
        
        recovery_action = {"action": "none", "details": ""}
        
        # Attempt retry if within limits
        if retry and step.retry_count < step.max_retries:
            step.retry_count += 1
            step.status = "pending"  # Reset for retry
            
            # Adjust approach for retry
            self._adapt_step_for_retry(step, failure_reason)
            
            recovery_action = {
                "action": "retry",
                "details": f"Retrying step {failed_step_id} (attempt {step.retry_count + 1})",
                "modified_step": True
            }
            
        # Try fallback if available
        elif step.fallback_step:
            fallback_step = self._find_step_by_id(plan, step.fallback_step)
            if fallback_step:
                fallback_step.status = "pending"
                step.status = "failed"
                
                recovery_action = {
                    "action": "fallback",
                    "details": f"Activating fallback step {step.fallback_step}",
                    "fallback_step_id": step.fallback_step
                }
        
        # Check branching alternatives
        elif failed_step_id in plan.branching_points:
            alternatives = plan.branching_points[failed_step_id]
            if alternatives:
                # Activate first alternative
                alt_step_id = f"{failed_step_id}_alt_{alternatives[0]}_0"
                alt_step = self._find_step_by_id(plan, alt_step_id)
                if alt_step:
                    alt_step.status = "pending"
                    
                    recovery_action = {
                        "action": "branch",
                        "details": f"Activating alternative path: {alternatives[0]}",
                        "alternative_step_id": alt_step_id
                    }
        
        # Last resort: mark as failed and continue
        else:
            step.status = "failed"
            recovery_action = {
                "action": "continue",
                "details": f"Step {failed_step_id} marked as failed, continuing with plan"
            }
        
        plan.updated_at = datetime.now().isoformat()
        
        if self.plan_memory:
            self.plan_memory.store_plan(plan)
        
        return recovery_action
    
    def adaptive_replan(self, plan: Plan, context_changes: Dict[str, Any]) -> Plan:
        """
        Adaptively modify a plan based on changing conditions.
        
        Args:
            plan: Plan to modify
            context_changes: Changes in context that require adaptation
            
        Returns:
            Modified plan or new plan if major changes needed
        """
        logging.info(f"[{self.name}] Adaptive re-planning for {plan.plan_id}")
        
        # Analyze the impact of context changes
        impact_analysis = self._analyze_context_impact(plan, context_changes)
        
        if impact_analysis["major_changes_needed"]:
            # Create a new plan incorporating lessons learned
            new_goal = f"{plan.goal_text} (adapted for {list(context_changes.keys())})"
            new_plan = self.plan_goal(new_goal, parent_goal_id=plan.parent_goal_id)
            
            # Transfer completed progress
            self._transfer_progress(plan, new_plan)
            
            # Mark old plan as superseded
            plan.status = "superseded"
            plan.updated_at = datetime.now().isoformat()
            
            return new_plan
        
        else:
            # Modify existing plan
            self._apply_adaptive_changes(plan, context_changes, impact_analysis)
            plan.updated_at = datetime.now().isoformat()
            
            if self.plan_memory:
                self.plan_memory.store_plan(plan)
            
            return plan
    
    def get_next_executable_step(self, plan: Plan) -> Optional[PlanStep]:
        """
        Get the next step that can be executed based on dependencies and status.
        
        Args:
            plan: Plan to analyze
            
        Returns:
            Next executable step or None if none available
        """
        executable_steps = []
        
        for step in plan.steps:
            if step.status != "pending":
                continue
                
            # Check if prerequisites are satisfied
            deps_satisfied = all(
                any(s.step_id == dep and s.status == "completed" for s in plan.steps)
                for dep in step.prerequisites
            )
            
            if deps_satisfied:
                executable_steps.append(step)
        
        if not executable_steps:
            return None
        
        # Return highest priority executable step
        return max(executable_steps, key=lambda s: s.priority)
    
    def visualize_plan_tree(self, plan: Plan) -> Dict[str, Any]:
        """
        Create a visualization structure for the plan tree.
        
        Args:
            plan: Plan to visualize
            
        Returns:
            Visualization data structure
        """
        nodes = []
        edges = []
        
        # Add nodes for each step
        for step in plan.steps:
            node = {
                "id": step.step_id,
                "label": step.subgoal,
                "status": step.status,
                "confidence": step.confidence,
                "priority": step.priority,
                "type": "step"
            }
            
            # Add status-specific styling
            if step.status == "completed":
                node["color"] = "green"
            elif step.status == "failed":
                node["color"] = "red"
            elif step.status == "in_progress":
                node["color"] = "yellow"
            else:
                node["color"] = "gray"
            
            nodes.append(node)
        
        # Add dependency edges
        for step in plan.steps:
            for prereq in step.prerequisites:
                edges.append({
                    "from": prereq,
                    "to": step.step_id,
                    "type": "dependency",
                    "color": "blue"
                })
        
        # Add next step edges
        for step in plan.steps:
            if step.next_step:
                edges.append({
                    "from": step.step_id,
                    "to": step.next_step,
                    "type": "sequence",
                    "color": "black"
                })
        
        # Add fallback edges
        for step in plan.steps:
            if step.fallback_step:
                edges.append({
                    "from": step.step_id,
                    "to": step.fallback_step,
                    "type": "fallback",
                    "color": "orange",
                    "style": "dashed"
                })
        
        # Add branching edges
        for step_id, alternatives in plan.branching_points.items():
            for alt in alternatives:
                alt_step_id = f"{step_id}_alt_{alt}_0"
                edges.append({
                    "from": step_id,
                    "to": alt_step_id,
                    "type": "branch",
                    "color": "purple",
                    "style": "dashed"
                })
        
        # Calculate layout hints
        levels = self._calculate_step_levels(plan)
        
        return {
            "plan_id": plan.plan_id,
            "goal": plan.goal_text,
            "status": plan.status,
            "nodes": nodes,
            "edges": edges,
            "levels": levels,
            "statistics": {
                "total_steps": len(plan.steps),
                "completed": len([s for s in plan.steps if s.status == "completed"]),
                "failed": len([s for s in plan.steps if s.status == "failed"]),
                "pending": len([s for s in plan.steps if s.status == "pending"]),
                "branches": len(plan.branching_points),
                "checkpoints": len(plan.progress_checkpoints)
            }
        }
    
    # === Helper Methods for Phase 16 Features ===
    
    def _find_step_by_id(self, plan: Plan, step_id: str) -> Optional[PlanStep]:
        """Find a step by ID within a plan."""
        for step in plan.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def _determine_plan_status(self, plan: Plan) -> str:
        """Determine overall plan status based on step statuses."""
        if not plan.steps:
            return "empty"
        
        statuses = [step.status for step in plan.steps]
        
        if all(status == "completed" for status in statuses):
            return "completed"
        elif any(status == "failed" for status in statuses):
            return "partially_failed"
        elif any(status == "in_progress" for status in statuses):
            return "in_progress"
        else:
            return "pending"
    
    def _generate_progress_recommendations(self, progress_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on progress analysis."""
        recommendations = []
        
        if progress_analysis["bottlenecks"]:
            recommendations.append("Consider breaking down bottleneck steps or adding parallel alternatives")
        
        if progress_analysis["blocked_steps"]:
            recommendations.append("Resolve dependency issues for blocked steps")
        
        if progress_analysis["failed_steps"] > 0:
            recommendations.append("Review and address failed steps before proceeding")
        
        if progress_analysis["progress_percentage"] < 25:
            recommendations.append("Plan execution is slow - consider resource allocation review")
        
        return recommendations
    
    def _adapt_step_for_retry(self, step: PlanStep, failure_reason: str) -> None:
        """Adapt a step for retry based on failure reason."""
        # Lower confidence for retry
        step.confidence = max(0.1, step.confidence * 0.8)
        
        # Adjust approach based on failure type
        if "timeout" in failure_reason.lower():
            # Increase estimated duration
            if step.estimated_duration:
                try:
                    current_duration = int(step.estimated_duration)
                    step.estimated_duration = str(int(current_duration * 1.5))
                except ValueError:
                    step.estimated_duration = "60"  # Default to 1 hour
        
        elif "resource" in failure_reason.lower():
            # Add resource requirement note
            step.why += f" (Retry: ensure resources available after previous failure)"
    
    def _analyze_context_impact(self, plan: Plan, context_changes: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how context changes impact the plan."""
        impact_score = 0.0
        affected_steps = []
        
        for change_type, change_value in context_changes.items():
            if change_type == "deadline_pressure":
                impact_score += change_value * 0.3
                # Mark steps that might need acceleration
                for step in plan.steps:
                    if step.status == "pending":
                        affected_steps.append(step.step_id)
            
            elif change_type == "resource_constraints":
                impact_score += change_value * 0.4
                # Mark resource-intensive steps
                for step in plan.steps:
                    if step.resources_needed:
                        affected_steps.append(step.step_id)
            
            elif change_type == "priority_shift":
                impact_score += change_value * 0.2
        
        return {
            "impact_score": impact_score,
            "major_changes_needed": impact_score > 0.6,
            "affected_steps": list(set(affected_steps)),
            "recommended_actions": self._recommend_adaptation_actions(impact_score, context_changes)
        }
    
    def _recommend_adaptation_actions(self, impact_score: float, context_changes: Dict[str, Any]) -> List[str]:
        """Recommend specific adaptation actions."""
        actions = []
        
        if impact_score > 0.8:
            actions.append("Complete re-planning recommended")
        elif impact_score > 0.6:
            actions.append("Significant plan modifications needed")
        elif impact_score > 0.3:
            actions.append("Minor adjustments to timeline and priorities")
        
        for change_type, value in context_changes.items():
            if change_type == "deadline_pressure" and value > 0.7:
                actions.append("Accelerate critical path steps")
            elif change_type == "resource_constraints" and value > 0.7:
                actions.append("Optimize resource usage or find alternatives")
        
        return actions
    
    def _apply_adaptive_changes(self, plan: Plan, context_changes: Dict[str, Any], impact_analysis: Dict[str, Any]) -> None:
        """Apply adaptive changes to a plan."""
        for step_id in impact_analysis["affected_steps"]:
            step = self._find_step_by_id(plan, step_id)
            if step:
                # Adjust priority based on context changes
                if "deadline_pressure" in context_changes:
                    step.priority = min(10, step.priority + int(context_changes["deadline_pressure"] * 3))
                
                # Adjust resource requirements
                if "resource_constraints" in context_changes and step.resources_needed:
                    # Add note about resource optimization
                    step.why += f" (Adapted: optimize resource usage due to constraints)"
    
    def _transfer_progress(self, old_plan: Plan, new_plan: Plan) -> None:
        """Transfer completed progress from old plan to new plan."""
        completed_goals = set()
        
        for step in old_plan.steps:
            if step.status == "completed":
                completed_goals.add(step.subgoal.lower())
        
        # Mark similar steps in new plan as completed
        for step in new_plan.steps:
            if step.subgoal.lower() in completed_goals:
                step.status = "completed"
                step.completed_at = datetime.now()
    
    def _calculate_step_levels(self, plan: Plan) -> Dict[str, int]:
        """Calculate hierarchical levels for visualization layout."""
        levels = {}
        
        # Steps with no prerequisites are at level 0
        for step in plan.steps:
            if not step.prerequisites:
                levels[step.step_id] = 0
        
        # Calculate levels for dependent steps
        changed = True
        while changed:
            changed = False
            for step in plan.steps:
                if step.step_id not in levels:
                    max_prereq_level = -1
                    all_prereqs_have_levels = True
                    
                    for prereq in step.prerequisites:
                        if prereq in levels:
                            max_prereq_level = max(max_prereq_level, levels[prereq])
                        else:
                            all_prereqs_have_levels = False
                            break
                    
                    if all_prereqs_have_levels:
                        levels[step.step_id] = max_prereq_level + 1
                        changed = True
        
        return levels