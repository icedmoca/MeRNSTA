#!/usr/bin/env python3
"""
DecisionPlanner - Autonomous Planning & Decision Layer for MeRNSTA

Generates multi-step plans, evaluates strategies, and manages adaptive re-planning.
Enables fully autonomous decision-making with sophisticated scoring and evaluation.
"""

import logging
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum

from .base import BaseAgent
from .recursive_planner import Plan, PlanStep


class StrategyType(Enum):
    """Types of planning strategies"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel" 
    TREE = "tree"
    DAG = "dag"
    CONDITIONAL = "conditional"
    ITERATIVE = "iterative"


class DecisionCriteria(Enum):
    """Criteria for decision evaluation"""
    FEASIBILITY = "feasibility"
    MEMORY_SIMILARITY = "memory_similarity"
    HISTORICAL_SUCCESS = "historical_success"
    ESTIMATED_EFFORT = "estimated_effort"
    RISK_LEVEL = "risk_level"
    URGENCY = "urgency"
    IMPACT = "impact"
    RESOURCE_AVAILABILITY = "resource_availability"


@dataclass
class PlanNode:
    """Enhanced plan node for DAG-based planning"""
    node_id: str
    step: PlanStep
    dependencies: Set[str]
    dependents: Set[str]
    conditions: List[str]
    fallback_nodes: List[str]
    retry_count: int = 0
    max_retries: int = 3
    estimated_duration: Optional[int] = None  # minutes
    resource_requirements: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.resource_requirements is None:
            self.resource_requirements = {}


@dataclass
class Strategy:
    """Complete strategy including plan and metadata"""
    strategy_id: str
    goal_text: str
    plan: Plan
    strategy_type: StrategyType
    nodes: List[PlanNode]
    score: float = 0.0
    scores_breakdown: Dict[str, float] = None
    created_at: datetime = None
    estimated_completion: datetime = None
    risk_assessment: Dict[str, Any] = None
    success_probability: float = 0.0
    
    def __post_init__(self):
        if self.scores_breakdown is None:
            self.scores_breakdown = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.risk_assessment is None:
            self.risk_assessment = {}


class DecisionPlanner(BaseAgent):
    """
    Autonomous planning and decision-making agent.
    
    Capabilities:
    - Multi-step plan generation with DAG support
    - Strategy scoring based on multiple criteria
    - Adaptive re-planning when conditions change
    - Integration with memory and reflection systems
    - Autonomous task selection and prioritization
    """
    
    def __init__(self):
        super().__init__("decision_planner")
        
        # Initialize planning configuration
        self.config = self._load_planning_config()
        self.enabled = self.config.get('enable_autonomous_planning', True)
        self.default_strategy_count = self.config.get('default_strategy_count', 3)
        
        # Scoring weights
        scoring_config = self.config.get('scoring', {})
        self.feasibility_weight = scoring_config.get('feasibility_weight', 1.0)
        self.memory_similarity_weight = scoring_config.get('memory_similarity_weight', 0.7)
        self.historical_success_weight = scoring_config.get('historical_success_weight', 1.2)
        self.effort_penalty = scoring_config.get('effort_penalty', 0.5)
        self.risk_penalty = scoring_config.get('risk_penalty', 0.8)
        self.urgency_boost = scoring_config.get('urgency_boost', 0.3)
        
        # Re-planning configuration
        self.replanning_threshold = self.config.get('replanning_threshold', 0.5)
        self.max_replanning_attempts = self.config.get('max_replanning_attempts', 3)
        
        # Initialize subsystems
        self._init_subsystems()
        
        # Strategy cache
        self.strategy_cache: Dict[str, Strategy] = {}
        self.active_strategies: Dict[str, Strategy] = {}
        
        logging.info(f"[{self.name}] Initialized with autonomous planning {'enabled' if self.enabled else 'disabled'}")
    
    def get_agent_instructions(self) -> str:
        """Return specialized instructions for the decision planner agent."""
        return (
            "You are a strategic decision planning specialist focused on goal achievement and strategy optimization. "
            "Your role is to analyze complex problems, generate multiple solution strategies, evaluate their feasibility, "
            "and select optimal approaches based on context, resources, and constraints. You coordinate task prioritization, "
            "manage execution plans, and adapt strategies based on outcomes. Focus on strategic thinking, planning efficiency, "
            "and adaptive decision-making to achieve objectives effectively."
        )
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """Generate strategic planning responses and coordinate decision-making."""
        context = context or {}
        
        # Build memory context for strategic patterns
        memory_context = self.get_memory_context(message)
        
        # Use LLM if available for complex strategic questions
        if self.llm_fallback:
            prompt = self.build_agent_prompt(message, memory_context)
            try:
                return self.llm_fallback.process(prompt)
            except Exception as e:
                logging.error(f"[{self.name}] LLM processing failed: {e}")
        
        # Handle planning and strategy requests
        if "plan" in message.lower() or "strategy" in message.lower():
            if "goal" in context:
                try:
                    strategies = self.generate_strategies(context["goal"])
                    best_strategy = self.select_best_strategy(context["goal"]) if strategies else None
                    if best_strategy:
                        return f"Generated {len(strategies)} strategies. Best strategy: {best_strategy.name} (score: {best_strategy.score:.2f})"
                    else:
                        return f"Generated {len(strategies)} strategies but couldn't select optimal one."
                except Exception as e:
                    return f"Strategy generation failed: {str(e)}"
            else:
                return "Please provide a goal to plan strategies for."
        
        if "next" in message.lower() and "action" in message.lower():
            try:
                next_action = self.get_next_action()
                if next_action:
                    return f"Next recommended action: {next_action.description} (priority: {next_action.priority})"
                else:
                    return "No pending actions in the current plan."
            except Exception as e:
                return f"Action selection failed: {str(e)}"
        
        return "I can help with strategic planning, goal analysis, and decision-making. Provide a goal or ask about planning strategies."

    def generate_strategies(self, goal_text: str, strategy_count: Optional[int] = None) -> List[Strategy]:
        """
        Generate multiple strategies for achieving a goal.
        
        Args:
            goal_text: The goal to plan for
            strategy_count: Number of strategies to generate
            
        Returns:
            List of generated strategies, sorted by score
        """
        if not self.enabled:
            logging.warning(f"[{self.name}] Autonomous planning disabled")
            return []
        
        strategy_count = strategy_count or self.default_strategy_count
        logging.info(f"[{self.name}] Generating {strategy_count} strategies for: {goal_text}")
        
        strategies = []
        
        # Generate different types of strategies
        strategy_types = [
            StrategyType.SEQUENTIAL,
            StrategyType.PARALLEL,
            StrategyType.TREE,
            StrategyType.CONDITIONAL,
            StrategyType.ITERATIVE
        ]
        
        for i in range(strategy_count):
            strategy_type = strategy_types[i % len(strategy_types)]
            
            try:
                strategy = self._generate_single_strategy(goal_text, strategy_type)
                if strategy:
                    strategies.append(strategy)
            except Exception as e:
                logging.error(f"[{self.name}] Error generating strategy {i}: {e}")
        
        # Score and rank strategies
        for strategy in strategies:
            strategy.score = self._score_strategy(strategy)
        
        # Sort by score (highest first)
        strategies.sort(key=lambda s: s.score, reverse=True)
        
        # Cache strategies
        for strategy in strategies:
            self.strategy_cache[strategy.strategy_id] = strategy
        
        logging.info(f"[{self.name}] Generated {len(strategies)} strategies, best score: {strategies[0].score:.3f}" if strategies else "No strategies generated")
        
        return strategies
    
    def select_best_strategy(self, goal_text: str) -> Optional[Strategy]:
        """
        Generate strategies and select the best one.
        
        Args:
            goal_text: The goal to plan for
            
        Returns:
            The highest-scoring strategy, or None if none generated
        """
        strategies = self.generate_strategies(goal_text)
        
        if not strategies:
            return None
        
        best_strategy = strategies[0]
        self.active_strategies[goal_text] = best_strategy
        
        logging.info(f"[{self.name}] Selected strategy {best_strategy.strategy_id} (score: {best_strategy.score:.3f}) for: {goal_text}")
        
        return best_strategy
    
    def adaptive_replan(self, failed_goal: str, failure_context: Dict[str, Any]) -> Optional[Strategy]:
        """
        Generate a new strategy after a failure.
        
        Args:
            failed_goal: The goal that failed
            failure_context: Context about what went wrong
            
        Returns:
            New strategy that accounts for the failure, or None
        """
        logging.info(f"[{self.name}] Adaptive re-planning for failed goal: {failed_goal}")
        
        # Analyze failure
        failure_analysis = self._analyze_failure(failed_goal, failure_context)
        
        # Generate new strategies that avoid the failure modes
        strategies = self._generate_failure_aware_strategies(failed_goal, failure_analysis)
        
        if not strategies:
            logging.warning(f"[{self.name}] Could not generate alternative strategies for: {failed_goal}")
            return None
        
        # Select best alternative
        best_strategy = strategies[0]
        self.active_strategies[failed_goal] = best_strategy
        
        # Log the re-planning
        self._log_replanning(failed_goal, failure_analysis, best_strategy)
        
        return best_strategy
    
    def get_next_action(self, goal_text: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the next action to execute from active strategies.
        
        Args:
            goal_text: Specific goal to get action for, or None for best overall action
            
        Returns:
            Next action to execute with context, or None
        """
        if goal_text:
            # Get next action for specific goal
            if goal_text in self.active_strategies:
                strategy = self.active_strategies[goal_text]
                return self._get_next_action_from_strategy(strategy)
            else:
                logging.warning(f"[{self.name}] No active strategy for goal: {goal_text}")
                return None
        else:
            # Get best next action across all active strategies
            return self._get_best_next_action()
    
    def update_strategy_progress(self, goal_text: str, step_id: str, status: str, 
                               result: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the progress of a strategy step.
        
        Args:
            goal_text: The goal being worked on
            step_id: The step that was executed
            status: New status (completed, failed, etc.)
            result: Execution result and context
        """
        if goal_text not in self.active_strategies:
            logging.warning(f"[{self.name}] No active strategy for goal: {goal_text}")
            return
        
        strategy = self.active_strategies[goal_text]
        
        # Update step status
        for node in strategy.nodes:
            if node.step.step_id == step_id:
                node.step.status = status
                break
        
        # Update plan steps
        for step in strategy.plan.steps:
            if step.step_id == step_id:
                step.status = status
                break
        
        # Log progress
        self._log_progress_update(goal_text, step_id, status, result)
        
        # Check if strategy needs re-planning
        if status == "failed" and self._should_replan(strategy):
            failure_context = {
                "failed_step": step_id,
                "result": result,
                "strategy_id": strategy.strategy_id
            }
            self.adaptive_replan(goal_text, failure_context)
    
    def visualize_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """
        Generate a visualization of a strategy's plan tree.
        
        Args:
            strategy_id: ID of the strategy to visualize
            
        Returns:
            Visualization data structure
        """
        if strategy_id not in self.strategy_cache:
            return {"error": "Strategy not found"}
        
        strategy = self.strategy_cache[strategy_id]
        
        visualization = {
            "strategy_id": strategy_id,
            "goal": strategy.goal_text,
            "type": strategy.strategy_type.value,
            "score": strategy.score,
            "nodes": [],
            "edges": [],
            "stats": {
                "total_steps": len(strategy.nodes),
                "completed_steps": sum(1 for node in strategy.nodes if node.step.status == "completed"),
                "failed_steps": sum(1 for node in strategy.nodes if node.step.status == "failed"),
                "estimated_duration": sum(node.estimated_duration or 0 for node in strategy.nodes)
            }
        }
        
        # Add nodes
        for node in strategy.nodes:
            visualization["nodes"].append({
                "id": node.node_id,
                "label": node.step.subgoal,
                "status": node.step.status,
                "confidence": node.step.confidence,
                "priority": node.step.priority,
                "dependencies": list(node.dependencies),
                "conditions": node.conditions,
                "retry_count": node.retry_count
            })
        
        # Add edges (dependencies)
        for node in strategy.nodes:
            for dep in node.dependencies:
                visualization["edges"].append({
                    "from": dep,
                    "to": node.node_id,
                    "type": "dependency"
                })
            
            for fallback in node.fallback_nodes:
                visualization["edges"].append({
                    "from": node.node_id,
                    "to": fallback,
                    "type": "fallback"
                })
        
        return visualization
    
    def _generate_single_strategy(self, goal_text: str, strategy_type: StrategyType) -> Optional[Strategy]:
        """Generate a single strategy of the specified type."""
        strategy_id = f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Use LLM or heuristics to decompose goal
        if self.llm_fallback:
            plan_data = self._llm_generate_plan(goal_text, strategy_type)
        else:
            plan_data = self._heuristic_generate_plan(goal_text, strategy_type)
        
        if not plan_data:
            return None
        
        # Create plan structure
        plan = Plan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            goal_text=goal_text,
            steps=plan_data["steps"],
            plan_type=strategy_type.value,
            created_at=datetime.now().isoformat(),
            status="draft",
            confidence=plan_data.get("confidence", 0.8),
            success_criteria=plan_data.get("success_criteria", []),
            risk_factors=plan_data.get("risk_factors", [])
        )
        
        # Create enhanced nodes for DAG support
        nodes = []
        for i, step in enumerate(plan.steps):
            node = PlanNode(
                node_id=f"node_{step.step_id}",
                step=step,
                dependencies=set(plan_data.get("dependencies", {}).get(step.step_id, [])),
                dependents=set(),
                conditions=plan_data.get("conditions", {}).get(step.step_id, []),
                fallback_nodes=plan_data.get("fallbacks", {}).get(step.step_id, []),
                estimated_duration=plan_data.get("durations", {}).get(step.step_id, 30)
            )
            nodes.append(node)
        
        # Set up dependent relationships
        for node in nodes:
            for dep_id in node.dependencies:
                for other_node in nodes:
                    if other_node.node_id == f"node_{dep_id}":
                        other_node.dependents.add(node.node_id)
        
        # Create strategy
        strategy = Strategy(
            strategy_id=strategy_id,
            goal_text=goal_text,
            plan=plan,
            strategy_type=strategy_type,
            nodes=nodes,
            estimated_completion=datetime.now() + timedelta(
                minutes=sum(node.estimated_duration for node in nodes)
            )
        )
        
        return strategy
    
    def _score_strategy(self, strategy: Strategy) -> float:
        """Score a strategy based on multiple criteria."""
        scores = {}
        
        # Feasibility score (0-1)
        scores[DecisionCriteria.FEASIBILITY.value] = self._calculate_feasibility_score(strategy)
        
        # Memory similarity score (0-1)
        scores[DecisionCriteria.MEMORY_SIMILARITY.value] = self._calculate_memory_similarity_score(strategy)
        
        # Historical success score (0-1)
        scores[DecisionCriteria.HISTORICAL_SUCCESS.value] = self._calculate_historical_success_score(strategy)
        
        # Effort penalty (0-1, higher effort = lower score)
        scores[DecisionCriteria.ESTIMATED_EFFORT.value] = self._calculate_effort_score(strategy)
        
        # Risk penalty (0-1, higher risk = lower score)
        scores[DecisionCriteria.RISK_LEVEL.value] = self._calculate_risk_score(strategy)
        
        # Urgency boost (0-1)
        scores[DecisionCriteria.URGENCY.value] = self._calculate_urgency_score(strategy)
        
        # Weighted combination
        total_score = (
            scores[DecisionCriteria.FEASIBILITY.value] * self.feasibility_weight +
            scores[DecisionCriteria.MEMORY_SIMILARITY.value] * self.memory_similarity_weight +
            scores[DecisionCriteria.HISTORICAL_SUCCESS.value] * self.historical_success_weight +
            scores[DecisionCriteria.ESTIMATED_EFFORT.value] * self.effort_penalty +
            scores[DecisionCriteria.RISK_LEVEL.value] * self.risk_penalty +
            scores[DecisionCriteria.URGENCY.value] * self.urgency_boost
        )
        
        # Normalize to 0-1 range
        max_possible_score = (
            self.feasibility_weight +
            self.memory_similarity_weight +
            self.historical_success_weight +
            self.effort_penalty +
            self.risk_penalty +
            self.urgency_boost
        )
        
        normalized_score = total_score / max_possible_score
        
        # Store breakdown
        strategy.scores_breakdown = scores
        strategy.success_probability = self._calculate_success_probability(strategy)
        
        return normalized_score
    
    def _llm_generate_plan(self, goal_text: str, strategy_type: StrategyType) -> Optional[Dict[str, Any]]:
        """Use LLM to generate a plan for the goal."""
        if not self.llm_fallback:
            return None
        
        prompt = f"""
        Create a detailed {strategy_type.value} plan for the following goal:
        Goal: {goal_text}
        
        Generate a plan with:
        1. 3-7 concrete, actionable steps
        2. Clear dependencies between steps
        3. Fallback options for high-risk steps
        4. Estimated duration for each step (in minutes)
        5. Success criteria and risk factors
        
        Return as JSON with this structure:
        {{
            "steps": [
                {{
                    "step_id": "step_1",
                    "subgoal": "specific action to take",
                    "why": "reasoning for this step",
                    "expected_result": "what should happen",
                    "prerequisites": ["list of prerequisites"],
                    "confidence": 0.8,
                    "priority": 1
                }}
            ],
            "dependencies": {{"step_1": ["step_0"], "step_2": ["step_1"]}},
            "conditions": {{"step_1": ["condition if needed"]}},
            "fallbacks": {{"step_1": ["fallback_step_id"]}},
            "durations": {{"step_1": 30, "step_2": 45}},
            "success_criteria": ["criteria for success"],
            "risk_factors": ["potential risks"],
            "confidence": 0.8
        }}
        """
        
        try:
            response = self.llm_fallback.process(prompt)
            plan_data = json.loads(response)
            return plan_data
        except Exception as e:
            logging.error(f"[{self.name}] LLM plan generation failed: {e}")
            return None
    
    def _heuristic_generate_plan(self, goal_text: str, strategy_type: StrategyType) -> Dict[str, Any]:
        """Generate a simple plan using heuristics."""
        # Simple fallback plan generation
        steps = []
        
        if "refactor" in goal_text.lower():
            steps = [
                PlanStep("step_1", "Analyze current code structure", "Understand existing implementation", "Code analysis complete", []),
                PlanStep("step_2", "Design new structure", "Plan the refactoring approach", "Design document ready", ["step_1"]),
                PlanStep("step_3", "Implement changes", "Execute the refactoring", "Code refactored", ["step_2"]),
                PlanStep("step_4", "Test changes", "Verify functionality", "Tests passing", ["step_3"]),
                PlanStep("step_5", "Deploy changes", "Roll out the changes", "Changes deployed", ["step_4"])
            ]
        elif "implement" in goal_text.lower() or "create" in goal_text.lower():
            steps = [
                PlanStep("step_1", "Research requirements", "Understand what needs to be built", "Requirements clear", []),
                PlanStep("step_2", "Design architecture", "Plan the implementation", "Architecture designed", ["step_1"]),
                PlanStep("step_3", "Implement core functionality", "Build the main features", "Core complete", ["step_2"]),
                PlanStep("step_4", "Add tests", "Ensure quality", "Tests written", ["step_3"]),
                PlanStep("step_5", "Documentation", "Document the implementation", "Documentation complete", ["step_4"])
            ]
        else:
            # Generic plan
            steps = [
                PlanStep("step_1", "Analyze the problem", "Understand what needs to be done", "Problem understood", []),
                PlanStep("step_2", "Plan approach", "Decide how to solve it", "Approach planned", ["step_1"]),
                PlanStep("step_3", "Execute plan", "Implement the solution", "Solution implemented", ["step_2"]),
                PlanStep("step_4", "Verify results", "Check if goal is achieved", "Results verified", ["step_3"])
            ]
        
        return {
            "steps": steps,
            "dependencies": {step.step_id: step.prerequisites for step in steps},
            "conditions": {},
            "fallbacks": {},
            "durations": {step.step_id: 30 for step in steps},
            "success_criteria": ["Goal achieved successfully"],
            "risk_factors": ["Unexpected complications"],
            "confidence": 0.7
        }
    
    def _calculate_feasibility_score(self, strategy: Strategy) -> float:
        """Calculate how feasible the strategy is."""
        # Consider factors like complexity, resource requirements, etc.
        complexity_score = 1.0 - (len(strategy.nodes) / 20)  # Simpler plans are more feasible
        confidence_score = sum(node.step.confidence for node in strategy.nodes) / len(strategy.nodes)
        
        return (complexity_score + confidence_score) / 2
    
    def _calculate_memory_similarity_score(self, strategy: Strategy) -> float:
        """Calculate similarity to successful past strategies."""
        # This would integrate with the memory system
        # For now, return a placeholder
        return 0.5
    
    def _calculate_historical_success_score(self, strategy: Strategy) -> float:
        """Calculate success rate based on historical data."""
        # This would analyze past performance of similar strategies
        # For now, return based on strategy type
        type_success_rates = {
            StrategyType.SEQUENTIAL: 0.8,
            StrategyType.PARALLEL: 0.6,
            StrategyType.TREE: 0.7,
            StrategyType.CONDITIONAL: 0.75,
            StrategyType.ITERATIVE: 0.85
        }
        return type_success_rates.get(strategy.strategy_type, 0.5)
    
    def _calculate_effort_score(self, strategy: Strategy) -> float:
        """Calculate effort required (lower effort = higher score)."""
        total_duration = sum(node.estimated_duration or 30 for node in strategy.nodes)
        # Normalize: 60 minutes = 1.0, 480 minutes (8 hours) = 0.0
        effort_score = max(0, 1.0 - (total_duration - 60) / 420)
        return effort_score
    
    def _calculate_risk_score(self, strategy: Strategy) -> float:
        """Calculate risk level (lower risk = higher score)."""
        risk_factors = len(strategy.plan.risk_factors or [])
        # More risk factors = lower score
        risk_score = max(0, 1.0 - (risk_factors / 10))
        return risk_score
    
    def _calculate_urgency_score(self, strategy: Strategy) -> float:
        """Calculate urgency boost."""
        # This would consider deadline pressure, priority, etc.
        # For now, return based on priority
        avg_priority = sum(node.step.priority for node in strategy.nodes) / len(strategy.nodes)
        return avg_priority / 10  # Assuming priority 1-10
    
    def _calculate_success_probability(self, strategy: Strategy) -> float:
        """Calculate overall success probability."""
        # Combine various factors
        confidence_score = sum(node.step.confidence for node in strategy.nodes) / len(strategy.nodes)
        feasibility_score = self._calculate_feasibility_score(strategy)
        historical_score = self._calculate_historical_success_score(strategy)
        
        return (confidence_score + feasibility_score + historical_score) / 3
    
    def _analyze_failure(self, failed_goal: str, failure_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze why a goal failed."""
        return {
            "failed_step": failure_context.get("failed_step"),
            "failure_type": self._categorize_failure(failure_context),
            "contributing_factors": self._identify_failure_factors(failure_context),
            "lessons_learned": self._extract_lessons(failure_context)
        }
    
    def _categorize_failure(self, failure_context: Dict[str, Any]) -> str:
        """Categorize the type of failure."""
        result = failure_context.get("result", {})
        
        if "timeout" in str(result).lower():
            return "timeout"
        elif "permission" in str(result).lower():
            return "permission_error"
        elif "resource" in str(result).lower():
            return "resource_unavailable"
        elif "dependency" in str(result).lower():
            return "dependency_failure"
        else:
            return "unknown"
    
    def _identify_failure_factors(self, failure_context: Dict[str, Any]) -> List[str]:
        """Identify factors that contributed to failure."""
        factors = []
        
        # Analyze the failure context
        if failure_context.get("failed_step"):
            factors.append(f"Step failure: {failure_context['failed_step']}")
        
        # Add more analysis based on context
        return factors
    
    def _extract_lessons(self, failure_context: Dict[str, Any]) -> List[str]:
        """Extract lessons learned from the failure."""
        lessons = []
        
        failure_type = self._categorize_failure(failure_context)
        
        if failure_type == "timeout":
            lessons.append("Increase time estimates for similar tasks")
        elif failure_type == "dependency_failure":
            lessons.append("Add better dependency checking")
        
        return lessons
    
    def _generate_failure_aware_strategies(self, failed_goal: str, failure_analysis: Dict[str, Any]) -> List[Strategy]:
        """Generate new strategies that account for the failure."""
        # Generate strategies with failure mitigation
        strategies = self.generate_strategies(failed_goal)
        
        # Modify strategies to avoid the failure mode
        for strategy in strategies:
            self._apply_failure_mitigation(strategy, failure_analysis)
        
        return strategies
    
    def _apply_failure_mitigation(self, strategy: Strategy, failure_analysis: Dict[str, Any]) -> None:
        """Apply failure mitigation to a strategy."""
        failure_type = failure_analysis.get("failure_type")
        
        if failure_type == "timeout":
            # Increase time estimates
            for node in strategy.nodes:
                if node.estimated_duration:
                    node.estimated_duration = int(node.estimated_duration * 1.5)
        
        elif failure_type == "dependency_failure":
            # Add more robust dependency checking
            for node in strategy.nodes:
                if node.dependencies:
                    node.conditions.append("Verify dependencies are available")
    
    def _get_next_action_from_strategy(self, strategy: Strategy) -> Optional[Dict[str, Any]]:
        """Get the next action from a specific strategy."""
        # Find ready-to-execute steps (dependencies satisfied)
        ready_nodes = []
        
        for node in strategy.nodes:
            if node.step.status == "pending":
                # Check if all dependencies are completed
                deps_satisfied = True
                for dep_id in node.dependencies:
                    dep_node = next((n for n in strategy.nodes if n.node_id == dep_id), None)
                    if not dep_node or dep_node.step.status != "completed":
                        deps_satisfied = False
                        break
                
                if deps_satisfied:
                    ready_nodes.append(node)
        
        if not ready_nodes:
            return None
        
        # Select highest priority ready node
        best_node = max(ready_nodes, key=lambda n: n.step.priority)
        
        return {
            "goal": strategy.goal_text,
            "strategy_id": strategy.strategy_id,
            "step_id": best_node.step.step_id,
            "action": best_node.step.subgoal,
            "why": best_node.step.why,
            "expected_result": best_node.step.expected_result,
            "confidence": best_node.step.confidence,
            "priority": best_node.step.priority,
            "estimated_duration": best_node.estimated_duration
        }
    
    def _get_best_next_action(self) -> Optional[Dict[str, Any]]:
        """Get the best next action across all active strategies."""
        all_actions = []
        
        for goal_text, strategy in self.active_strategies.items():
            action = self._get_next_action_from_strategy(strategy)
            if action:
                action["strategy_score"] = strategy.score
                all_actions.append(action)
        
        if not all_actions:
            return None
        
        # Select action with highest combined score
        def action_score(action):
            return action["priority"] * action["confidence"] * action["strategy_score"]
        
        return max(all_actions, key=action_score)
    
    def _should_replan(self, strategy: Strategy) -> bool:
        """Determine if a strategy should be re-planned."""
        failed_steps = sum(1 for node in strategy.nodes if node.step.status == "failed")
        total_steps = len(strategy.nodes)
        
        failure_rate = failed_steps / total_steps if total_steps > 0 else 0
        
        return failure_rate > self.replanning_threshold
    
    def _log_replanning(self, failed_goal: str, failure_analysis: Dict[str, Any], new_strategy: Strategy) -> None:
        """Log re-planning activity."""
        logging.info(f"[{self.name}] Re-planned goal '{failed_goal}' -> new strategy {new_strategy.strategy_id}")
        logging.debug(f"[{self.name}] Failure analysis: {failure_analysis}")
    
    def _log_progress_update(self, goal_text: str, step_id: str, status: str, result: Optional[Dict[str, Any]]) -> None:
        """Log progress updates."""
        logging.info(f"[{self.name}] Progress update for '{goal_text}': {step_id} -> {status}")
        if result:
            logging.debug(f"[{self.name}] Result: {result}")
    
    def _init_subsystems(self) -> None:
        """Initialize subsystem connections."""
        try:
            # Initialize plan memory connection
            from storage.plan_memory import PlanMemory
            self.plan_memory = PlanMemory()
        except ImportError:
            logging.warning(f"[{self.name}] Plan memory not available")
            self.plan_memory = None
        
        try:
            # Initialize strategy evaluator
            from .strategy_evaluator import StrategyEvaluator
            self.strategy_evaluator = StrategyEvaluator()
        except ImportError:
            logging.warning(f"[{self.name}] Strategy evaluator not available")
            self.strategy_evaluator = None
        
        try:
            # Initialize task selector
            from .task_selector import TaskSelector
            self.task_selector = TaskSelector()
        except ImportError:
            logging.warning(f"[{self.name}] Task selector not available")
            self.task_selector = None
    
    def _load_planning_config(self) -> Dict[str, Any]:
        """Load planning configuration."""
        from config.settings import get_config
        
        config = get_config().get('decision_planner', {})
        
        # Default configuration
        default_config = {
            'enable_autonomous_planning': True,
            'default_strategy_count': 3,
            'scoring': {
                'feasibility_weight': 1.0,
                'memory_similarity_weight': 0.7,
                'historical_success_weight': 1.2,
                'effort_penalty': 0.5,
                'risk_penalty': 0.8,
                'urgency_boost': 0.3
            },
            'replanning_threshold': 0.5,
            'max_replanning_attempts': 3
        }
        
        # Merge with user config
        default_config.update(config)
        return default_config