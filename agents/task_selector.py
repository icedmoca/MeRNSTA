#!/usr/bin/env python3
"""
TaskSelector - Intelligent Task Selection and Prioritization for MeRNSTA

Autonomously selects which goals to pursue next based on urgency, feasibility,
historical impact, resource availability, and strategic alignment.
"""

import logging
import json
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .base import BaseAgent


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


class TaskCategory(Enum):
    """Categories of tasks"""
    MAINTENANCE = "maintenance"
    IMPROVEMENT = "improvement"
    RESEARCH = "research"
    REPAIR = "repair"
    OPTIMIZATION = "optimization"
    LEARNING = "learning"
    SOCIAL = "social"
    CREATIVE = "creative"
    STRATEGIC = "strategic"


@dataclass
class Task:
    """Represents a task or goal to be evaluated"""
    task_id: str
    goal_text: str
    category: TaskCategory
    priority: TaskPriority
    urgency: float  # 0-1
    importance: float  # 0-1
    feasibility: float  # 0-1
    estimated_effort: int  # minutes
    deadline: Optional[datetime] = None
    dependencies: List[str] = None
    resource_requirements: Dict[str, Any] = None
    context: Dict[str, Any] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.resource_requirements is None:
            self.resource_requirements = {}
        if self.context is None:
            self.context = {}
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class SelectionCriteria:
    """Criteria for task selection"""
    urgency_weight: float = 1.0
    importance_weight: float = 1.2
    feasibility_weight: float = 0.8
    impact_weight: float = 1.0
    effort_penalty: float = 0.3
    deadline_pressure_weight: float = 1.5
    dependency_bonus: float = 0.2
    category_preferences: Dict[str, float] = None
    
    def __post_init__(self):
        if self.category_preferences is None:
            self.category_preferences = {}


@dataclass
class SelectionResult:
    """Result of task selection"""
    selected_task: Task
    selection_score: float
    rationale: str
    alternative_tasks: List[Tuple[Task, float]]
    selection_timestamp: datetime
    context_factors: Dict[str, Any]


class TaskSelector(BaseAgent):
    """
    Intelligent task selection and prioritization system.
    
    Capabilities:
    - Multi-criteria task evaluation and ranking
    - Dynamic priority adjustment based on context
    - Deadline and dependency management
    - Resource availability consideration
    - Historical impact analysis
    - Strategic alignment assessment
    """
    
    def __init__(self):
        super().__init__("task_selector")
        
        # Load configuration
        self.config = self._load_selector_config()
        self.enabled = self.config.get('enabled', True)
        
        # Selection criteria
        self.base_criteria = SelectionCriteria(
            urgency_weight=self.config.get('urgency_weight', 1.0),
            importance_weight=self.config.get('importance_weight', 1.2),
            feasibility_weight=self.config.get('feasibility_weight', 0.8),
            impact_weight=self.config.get('impact_weight', 1.0),
            effort_penalty=self.config.get('effort_penalty', 0.3),
            deadline_pressure_weight=self.config.get('deadline_pressure_weight', 1.5),
            dependency_bonus=self.config.get('dependency_bonus', 0.2),
            category_preferences=self.config.get('category_preferences', {})
        )
        
        # State tracking
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.task_history: Dict[str, Dict[str, Any]] = {}
        self.resource_state: Dict[str, Any] = {}
        self.strategic_focus: List[str] = []
        
        # Performance tracking
        self.selection_history: List[SelectionResult] = []
        self.impact_history: Dict[str, float] = {}
        
        logging.info(f"[{self.name}] Initialized task selector")
    
    def get_agent_instructions(self) -> str:
        """Return specialized instructions for the task selector agent."""
        return (
            "You are a task selection and prioritization specialist focused on optimal task management. "
            "Your role is to analyze task queues, evaluate priorities based on urgency, importance, deadlines, "
            "and resource constraints. You coordinate task dependencies, manage workflow optimization, and "
            "ensure strategic alignment with goals. Focus on efficient resource allocation, deadline management, "
            "and maximizing overall productivity through intelligent task selection and sequencing."
        )
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """Generate task selection responses and coordinate task management."""
        context = context or {}
        
        # Build memory context for task patterns
        memory_context = self.get_memory_context(message)
        
        # Use LLM if available for complex task management questions
        if self.llm_fallback:
            prompt = self.build_agent_prompt(message, memory_context)
            try:
                return self.llm_fallback.process(prompt)
            except Exception as e:
                logging.error(f"[{self.name}] LLM processing failed: {e}")
        
        # Handle task management requests
        if "select" in message.lower() and "task" in message.lower():
            try:
                result = self.select_next_task(context.get("criteria"))
                if result and result.selected_task:
                    return f"Selected task: {result.selected_task.goal_text} (score: {result.selection_score:.2f})"
                else:
                    return "No suitable task found in the queue."
            except Exception as e:
                return f"Task selection failed: {str(e)}"
        
        if "add" in message.lower() and "task" in message.lower():
            if "goal" in context:
                try:
                    task_id = self.add_task(
                        context["goal"],
                        category=context.get("category", TaskCategory.IMPROVEMENT),
                        priority=context.get("priority", TaskPriority.MEDIUM)
                    )
                    return f"Added task with ID: {task_id}"
                except Exception as e:
                    return f"Failed to add task: {str(e)}"
            else:
                return "Please provide a goal for the task."
        
        if "queue" in message.lower() or "status" in message.lower():
            queue_size = len(self.task_queue)
            completed_count = len(self.completed_tasks)
            return f"Task queue status: {queue_size} pending tasks, {completed_count} completed tasks."
        
        return "I can help with task selection, prioritization, and queue management. Add tasks, request selection, or ask about queue status."

    def add_task(self, goal_text: str, category: TaskCategory = TaskCategory.IMPROVEMENT,
                priority: TaskPriority = TaskPriority.MEDIUM, urgency: float = 0.5,
                importance: float = 0.5, estimated_effort: int = 60,
                deadline: Optional[datetime] = None, **kwargs) -> str:
        """
        Add a new task to the selection queue.
        
        Args:
            goal_text: Description of the goal/task
            category: Category of the task
            priority: Initial priority level
            urgency: Urgency score (0-1)
            importance: Importance score (0-1)
            estimated_effort: Estimated effort in minutes
            deadline: Optional deadline
            **kwargs: Additional context and parameters
            
        Returns:
            Task ID
        """
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.task_queue):03d}"
        
        # Calculate feasibility
        feasibility = self._estimate_feasibility(goal_text, category, kwargs)
        
        task = Task(
            task_id=task_id,
            goal_text=goal_text,
            category=category,
            priority=priority,
            urgency=urgency,
            importance=importance,
            feasibility=feasibility,
            estimated_effort=estimated_effort,
            deadline=deadline,
            dependencies=kwargs.get('dependencies', []),
            resource_requirements=kwargs.get('resource_requirements', {}),
            context=kwargs
        )
        
        self.task_queue.append(task)
        logging.info(f"[{self.name}] Added task {task_id}: {goal_text}")
        
        return task_id
    
    def select_next_task(self, context: Optional[Dict[str, Any]] = None) -> Optional[SelectionResult]:
        """
        Select the next task to execute based on current context.
        
        Args:
            context: Current system context (resources, time constraints, etc.)
            
        Returns:
            Selection result with chosen task and rationale
        """
        if not self.task_queue:
            logging.info(f"[{self.name}] No tasks in queue")
            return None
        
        context = context or {}
        logging.info(f"[{self.name}] Selecting from {len(self.task_queue)} available tasks")
        
        # Filter tasks that are ready to execute
        ready_tasks = self._filter_ready_tasks(context)
        
        if not ready_tasks:
            logging.warning(f"[{self.name}] No tasks are ready for execution")
            return None
        
        # Score all ready tasks
        task_scores = []
        for task in ready_tasks:
            score = self._calculate_task_score(task, context)
            task_scores.append((task, score))
        
        # Sort by score (highest first)
        task_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_task, selection_score = task_scores[0]
        alternatives = task_scores[1:6]  # Top 5 alternatives
        
        # Generate selection rationale
        rationale = self._generate_selection_rationale(selected_task, selection_score, context)
        
        # Create selection result
        result = SelectionResult(
            selected_task=selected_task,
            selection_score=selection_score,
            rationale=rationale,
            alternative_tasks=alternatives,
            selection_timestamp=datetime.now(),
            context_factors=context
        )
        
        # Update state
        self._update_selection_state(result)
        
        return result
    
    def update_task_status(self, task_id: str, status: str, outcome: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the status of a task after execution.
        
        Args:
            task_id: ID of the task
            status: New status (completed, failed, postponed, etc.)
            outcome: Execution outcome and results
        """
        # Find and update task
        task = None
        for t in self.task_queue:
            if t.task_id == task_id:
                task = t
                break
        
        if not task:
            logging.warning(f"[{self.name}] Task {task_id} not found")
            return
        
        # Update task history
        self.task_history[task_id] = {
            'status': status,
            'outcome': outcome,
            'completed_at': datetime.now(),
            'duration': outcome.get('duration_minutes', 0) if outcome else 0
        }
        
        # Move completed tasks
        if status == 'completed':
            self.task_queue.remove(task)
            self.completed_tasks.append(task)
            
            # Update impact history
            if outcome:
                impact_score = self._calculate_impact_score(task, outcome)
                self.impact_history[task_id] = impact_score
        
        # Learn from the outcome
        self._learn_from_outcome(task, status, outcome)
        
        logging.info(f"[{self.name}] Updated task {task_id} status to {status}")
    
    def reorder_tasks(self, new_priorities: Optional[Dict[str, float]] = None) -> None:
        """
        Reorder tasks based on changing priorities or context.
        
        Args:
            new_priorities: Optional mapping of task_id to new priority scores
        """
        if new_priorities:
            # Update task priorities
            for task in self.task_queue:
                if task.task_id in new_priorities:
                    # Convert priority score to urgency/importance
                    priority_score = new_priorities[task.task_id]
                    task.urgency = min(1.0, task.urgency + (priority_score - 0.5) * 0.2)
                    task.importance = min(1.0, task.importance + (priority_score - 0.5) * 0.2)
        
        # Re-calculate priorities based on current context
        current_context = self._get_current_context()
        
        for task in self.task_queue:
            # Adjust urgency based on deadline pressure
            if task.deadline:
                time_to_deadline = (task.deadline - datetime.now()).total_seconds() / 3600  # hours
                if time_to_deadline < 24:  # Less than 1 day
                    task.urgency = min(1.0, task.urgency + 0.3)
                elif time_to_deadline < 168:  # Less than 1 week
                    task.urgency = min(1.0, task.urgency + 0.1)
            
            # Adjust importance based on strategic focus
            if any(focus in task.goal_text.lower() for focus in self.strategic_focus):
                task.importance = min(1.0, task.importance + 0.2)
        
        logging.info(f"[{self.name}] Reordered {len(self.task_queue)} tasks")
    
    def get_task_queue_status(self) -> Dict[str, Any]:
        """
        Get current status of the task queue.
        
        Returns:
            Status information about the task queue
        """
        if not self.task_queue:
            return {
                "queue_size": 0,
                "ready_tasks": 0,
                "message": "No tasks in queue"
            }
        
        # Analyze queue composition
        category_counts = {}
        priority_counts = {}
        ready_count = 0
        
        current_context = self._get_current_context()
        ready_tasks = self._filter_ready_tasks(current_context)
        ready_count = len(ready_tasks)
        
        for task in self.task_queue:
            category_counts[task.category.value] = category_counts.get(task.category.value, 0) + 1
            priority_counts[task.priority.value] = priority_counts.get(task.priority.value, 0) + 1
        
        # Calculate urgency distribution
        urgency_scores = [task.urgency for task in self.task_queue]
        avg_urgency = sum(urgency_scores) / len(urgency_scores) if urgency_scores else 0
        
        # Find tasks with approaching deadlines
        upcoming_deadlines = []
        now = datetime.now()
        for task in self.task_queue:
            if task.deadline and (task.deadline - now).total_seconds() < 86400:  # 24 hours
                upcoming_deadlines.append({
                    "task_id": task.task_id,
                    "goal": task.goal_text,
                    "deadline": task.deadline.isoformat(),
                    "hours_remaining": (task.deadline - now).total_seconds() / 3600
                })
        
        return {
            "queue_size": len(self.task_queue),
            "ready_tasks": ready_count,
            "completed_tasks": len(self.completed_tasks),
            "category_distribution": category_counts,
            "priority_distribution": priority_counts,
            "average_urgency": avg_urgency,
            "upcoming_deadlines": upcoming_deadlines,
            "strategic_focus": self.strategic_focus
        }
    
    def get_selection_insights(self) -> Dict[str, Any]:
        """
        Get insights about task selection patterns and performance.
        
        Returns:
            Analysis of selection patterns and outcomes
        """
        if not self.selection_history:
            return {"message": "No selection history available"}
        
        # Analyze selection patterns
        category_selections = {}
        priority_selections = {}
        
        for result in self.selection_history:
            category = result.selected_task.category.value
            priority = result.selected_task.priority.value
            
            category_selections[category] = category_selections.get(category, 0) + 1
            priority_selections[priority] = priority_selections.get(priority, 0) + 1
        
        # Calculate average scores
        avg_selection_score = sum(r.selection_score for r in self.selection_history) / len(self.selection_history)
        
        # Analyze impact correlation
        impact_correlation = self._analyze_impact_correlation()
        
        return {
            "total_selections": len(self.selection_history),
            "average_selection_score": avg_selection_score,
            "category_preferences": category_selections,
            "priority_preferences": priority_selections,
            "impact_correlation": impact_correlation,
            "selection_accuracy": self._calculate_selection_accuracy(),
            "recommendations": self._generate_selector_recommendations()
        }
    
    def set_strategic_focus(self, focus_areas: List[str]) -> None:
        """
        Set strategic focus areas to influence task selection.
        
        Args:
            focus_areas: List of keywords or themes to prioritize
        """
        self.strategic_focus = focus_areas
        logging.info(f"[{self.name}] Set strategic focus: {', '.join(focus_areas)}")
        
        # Reorder tasks based on new focus
        self.reorder_tasks()
    
    def _filter_ready_tasks(self, context: Dict[str, Any]) -> List[Task]:
        """Filter tasks that are ready for execution."""
        ready_tasks = []
        
        for task in self.task_queue:
            # Check dependencies
            if not self._dependencies_satisfied(task):
                continue
            
            # Check resource availability
            if not self._resources_available(task, context):
                continue
            
            # Check deadline constraints
            if not self._deadline_feasible(task):
                continue
            
            ready_tasks.append(task)
        
        return ready_tasks
    
    def _dependencies_satisfied(self, task: Task) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in self.task_history or self.task_history[dep_id]['status'] != 'completed':
                return False
        return True
    
    def _resources_available(self, task: Task, context: Dict[str, Any]) -> bool:
        """Check if required resources are available."""
        available_resources = context.get('available_resources', {})
        
        for resource, required_amount in task.resource_requirements.items():
            available_amount = available_resources.get(resource, float('inf'))
            if available_amount < required_amount:
                return False
        
        return True
    
    def _deadline_feasible(self, task: Task) -> bool:
        """Check if task can be completed before deadline."""
        if not task.deadline:
            return True
        
        time_available = (task.deadline - datetime.now()).total_seconds() / 60  # minutes
        return time_available >= task.estimated_effort
    
    def _calculate_task_score(self, task: Task, context: Dict[str, Any]) -> float:
        """Calculate a comprehensive score for task selection."""
        criteria = self._adapt_criteria_to_context(context)
        
        # Base scores
        urgency_score = task.urgency
        importance_score = task.importance
        feasibility_score = task.feasibility
        
        # Impact score (based on historical data)
        impact_score = self._estimate_impact_score(task)
        
        # Effort penalty (higher effort = lower score)
        effort_score = max(0, 1.0 - (task.estimated_effort / 480))  # 8 hours max
        
        # Deadline pressure
        deadline_score = self._calculate_deadline_pressure(task)
        
        # Dependency bonus (tasks that unlock others get bonus)
        dependency_score = self._calculate_dependency_bonus(task)
        
        # Category preference
        category_score = criteria.category_preferences.get(task.category.value, 0.5)
        
        # Strategic alignment
        strategic_score = self._calculate_strategic_alignment(task)
        
        # Weighted combination
        total_score = (
            urgency_score * criteria.urgency_weight +
            importance_score * criteria.importance_weight +
            feasibility_score * criteria.feasibility_weight +
            impact_score * criteria.impact_weight +
            effort_score * criteria.effort_penalty +
            deadline_score * criteria.deadline_pressure_weight +
            dependency_score * criteria.dependency_bonus +
            category_score * 0.3 +
            strategic_score * 0.4
        )
        
        # Normalize
        max_possible = (
            criteria.urgency_weight +
            criteria.importance_weight +
            criteria.feasibility_weight +
            criteria.impact_weight +
            criteria.effort_penalty +
            criteria.deadline_pressure_weight +
            criteria.dependency_bonus +
            0.3 + 0.4  # category and strategic weights
        )
        
        normalized_score = total_score / max_possible
        
        return min(1.0, max(0.0, normalized_score))
    
    def _estimate_feasibility(self, goal_text: str, category: TaskCategory, context: Dict[str, Any]) -> float:
        """Estimate how feasible a task is."""
        # Base feasibility by category
        category_feasibility = {
            TaskCategory.MAINTENANCE: 0.9,
            TaskCategory.REPAIR: 0.8,
            TaskCategory.OPTIMIZATION: 0.7,
            TaskCategory.IMPROVEMENT: 0.6,
            TaskCategory.LEARNING: 0.7,
            TaskCategory.RESEARCH: 0.5,
            TaskCategory.CREATIVE: 0.4,
            TaskCategory.STRATEGIC: 0.5,
            TaskCategory.SOCIAL: 0.6
        }
        
        base_feasibility = category_feasibility.get(category, 0.5)
        
        # Adjust based on goal complexity
        if len(goal_text.split()) > 20:  # Complex description
            base_feasibility *= 0.8
        
        # Adjust based on historical success with similar tasks
        historical_adjustment = self._get_historical_feasibility(goal_text, category)
        
        return (base_feasibility + historical_adjustment) / 2
    
    def _estimate_impact_score(self, task: Task) -> float:
        """Estimate the potential impact of completing this task."""
        # Base impact by category
        category_impact = {
            TaskCategory.STRATEGIC: 0.9,
            TaskCategory.IMPROVEMENT: 0.8,
            TaskCategory.OPTIMIZATION: 0.7,
            TaskCategory.REPAIR: 0.8,
            TaskCategory.LEARNING: 0.6,
            TaskCategory.MAINTENANCE: 0.5,
            TaskCategory.RESEARCH: 0.6,
            TaskCategory.CREATIVE: 0.5,
            TaskCategory.SOCIAL: 0.4
        }
        
        base_impact = category_impact.get(task.category, 0.5)
        
        # Adjust based on historical impact of similar tasks
        historical_impact = self._get_historical_impact(task)
        
        # Adjust based on effort (sometimes higher effort = higher impact)
        effort_factor = min(1.0, task.estimated_effort / 240)  # 4 hours reference
        
        return (base_impact + historical_impact + effort_factor * 0.2) / 2.2
    
    def _calculate_deadline_pressure(self, task: Task) -> float:
        """Calculate pressure score based on deadline proximity."""
        if not task.deadline:
            return 0.5  # Neutral score for tasks without deadlines
        
        time_remaining = (task.deadline - datetime.now()).total_seconds() / 3600  # hours
        
        if time_remaining <= 0:
            return 1.0  # Maximum pressure for overdue tasks
        elif time_remaining <= 4:
            return 0.9  # High pressure for tasks due within 4 hours
        elif time_remaining <= 24:
            return 0.7  # Medium-high pressure for tasks due within 1 day
        elif time_remaining <= 168:
            return 0.4  # Medium pressure for tasks due within 1 week
        else:
            return 0.2  # Low pressure for tasks due later
    
    def _calculate_dependency_bonus(self, task: Task) -> float:
        """Calculate bonus for tasks that unblock other tasks."""
        # Count how many tasks depend on this one
        dependent_count = 0
        for other_task in self.task_queue:
            if task.task_id in other_task.dependencies:
                dependent_count += 1
        
        # Bonus increases with number of dependent tasks
        return min(1.0, dependent_count * 0.2)
    
    def _calculate_strategic_alignment(self, task: Task) -> float:
        """Calculate how well the task aligns with strategic focus."""
        if not self.strategic_focus:
            return 0.5  # Neutral if no strategic focus set
        
        goal_lower = task.goal_text.lower()
        alignment_score = 0.0
        
        for focus_area in self.strategic_focus:
            if focus_area.lower() in goal_lower:
                alignment_score += 0.3
        
        return min(1.0, alignment_score)
    
    def _adapt_criteria_to_context(self, context: Dict[str, Any]) -> SelectionCriteria:
        """Adapt selection criteria based on current context."""
        criteria = SelectionCriteria(
            urgency_weight=self.base_criteria.urgency_weight,
            importance_weight=self.base_criteria.importance_weight,
            feasibility_weight=self.base_criteria.feasibility_weight,
            impact_weight=self.base_criteria.impact_weight,
            effort_penalty=self.base_criteria.effort_penalty,
            deadline_pressure_weight=self.base_criteria.deadline_pressure_weight,
            dependency_bonus=self.base_criteria.dependency_bonus,
            category_preferences=self.base_criteria.category_preferences.copy()
        )
        
        # Adapt based on context
        time_pressure = context.get('time_pressure', 0.5)
        if time_pressure > 0.7:
            criteria.urgency_weight *= 1.5
            criteria.deadline_pressure_weight *= 1.3
            criteria.feasibility_weight *= 1.2
        
        resource_constraints = context.get('resource_constraints', 0.5)
        if resource_constraints > 0.7:
            criteria.effort_penalty *= 1.4
            criteria.feasibility_weight *= 1.3
        
        return criteria
    
    def _get_current_context(self) -> Dict[str, Any]:
        """Get current system context."""
        return {
            'available_resources': self.resource_state,
            'time_pressure': 0.5,  # Would be calculated based on deadlines
            'resource_constraints': 0.3,  # Would be calculated based on availability
            'strategic_focus_active': len(self.strategic_focus) > 0
        }
    
    def _get_historical_feasibility(self, goal_text: str, category: TaskCategory) -> float:
        """Get historical feasibility for similar tasks."""
        # This would analyze historical data
        # For now, return a neutral value
        return 0.5
    
    def _get_historical_impact(self, task: Task) -> float:
        """Get historical impact for similar tasks."""
        # This would analyze historical impact data
        # For now, return based on category
        return 0.5
    
    def _calculate_impact_score(self, task: Task, outcome: Dict[str, Any]) -> float:
        """Calculate actual impact score from task outcome."""
        # This would analyze the outcome to determine impact
        success = outcome.get('success', False)
        duration = outcome.get('duration_minutes', task.estimated_effort)
        quality = outcome.get('quality_score', 0.5)
        
        # Simple impact calculation
        impact = 0.0
        if success:
            impact += 0.5
        
        # Efficiency bonus (completed faster than expected)
        if duration < task.estimated_effort:
            impact += 0.2
        
        # Quality bonus
        impact += quality * 0.3
        
        return min(1.0, impact)
    
    def _generate_selection_rationale(self, task: Task, score: float, context: Dict[str, Any]) -> str:
        """Generate human-readable rationale for task selection."""
        rationale_parts = [
            f"Selected '{task.goal_text}' with score {score:.3f}",
            f"Category: {task.category.value}, Priority: {task.priority.value}",
            f"Urgency: {task.urgency:.2f}, Importance: {task.importance:.2f}"
        ]
        
        if task.deadline:
            time_to_deadline = (task.deadline - datetime.now()).total_seconds() / 3600
            rationale_parts.append(f"Deadline in {time_to_deadline:.1f} hours")
        
        if self.strategic_focus and any(focus.lower() in task.goal_text.lower() for focus in self.strategic_focus):
            rationale_parts.append("Aligns with strategic focus")
        
        return ". ".join(rationale_parts)
    
    def _update_selection_state(self, result: SelectionResult) -> None:
        """Update internal state after task selection."""
        self.selection_history.append(result)
        
        # Update resource state (would be more sophisticated in practice)
        selected_task = result.selected_task
        for resource, amount in selected_task.resource_requirements.items():
            current_amount = self.resource_state.get(resource, float('inf'))
            self.resource_state[resource] = max(0, current_amount - amount)
    
    def _learn_from_outcome(self, task: Task, status: str, outcome: Optional[Dict[str, Any]]) -> None:
        """Learn from task execution outcomes."""
        # Update feasibility estimates
        if outcome:
            actual_duration = outcome.get('duration_minutes', task.estimated_effort)
            success = outcome.get('success', False)
            
            # Learn about estimation accuracy
            if actual_duration != task.estimated_effort:
                estimation_error = abs(actual_duration - task.estimated_effort) / task.estimated_effort
                # Would update estimation models here
                
        logging.debug(f"[{self.name}] Learning from outcome: {task.task_id} -> {status}")
    
    def _analyze_impact_correlation(self) -> Dict[str, float]:
        """Analyze correlation between selection scores and actual impact."""
        if len(self.impact_history) < 5:
            return {"correlation": 0.0, "confidence": 0.0}
        
        # Would calculate actual correlation
        # For now, return placeholder
        return {"correlation": 0.7, "confidence": 0.8}
    
    def _calculate_selection_accuracy(self) -> float:
        """Calculate how accurate task selections have been."""
        if not self.selection_history:
            return 0.0
        
        # Would analyze actual vs predicted outcomes
        # For now, return placeholder
        return 0.75
    
    def _generate_selector_recommendations(self) -> List[str]:
        """Generate recommendations for improving task selection."""
        recommendations = []
        
        if len(self.selection_history) < 10:
            recommendations.append("Gather more selection data for better analysis")
        
        if not self.strategic_focus:
            recommendations.append("Set strategic focus areas to improve selection quality")
        
        # Add more sophisticated recommendations based on analysis
        return recommendations
    
    def _load_selector_config(self) -> Dict[str, Any]:
        """Load task selector configuration."""
        from config.settings import get_config
        
        config = get_config().get('task_selector', {})
        
        # Default configuration
        default_config = {
            'enabled': True,
            'urgency_weight': 1.0,
            'importance_weight': 1.2,
            'feasibility_weight': 0.8,
            'impact_weight': 1.0,
            'effort_penalty': 0.3,
            'deadline_pressure_weight': 1.5,
            'dependency_bonus': 0.2,
            'category_preferences': {
                'maintenance': 0.7,
                'repair': 0.8,
                'improvement': 0.6,
                'strategic': 0.9
            }
        }
        
        # Merge with user config
        default_config.update(config)
        return default_config