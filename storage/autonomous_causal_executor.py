#!/usr/bin/env python3
"""
🔄 Autonomous Causal Executor for MeRNSTA

Automatically executes causal meta-goals by:
- Running clarification queries
- Processing user responses
- Updating memory based on feedback
- Creating feedback loops for continuous improvement
"""

import logging
import time
import asyncio
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
from collections import deque
import threading

from .causal_meta_goal_generator import CausalMetaGoalGenerator, CausalGoal
from .phase2_cognitive_system import Phase2AutonomousCognitiveSystem


@dataclass
class ExecutionResult:
    """Result of executing a causal goal."""
    goal_id: str
    query: str
    response: str
    confidence_change: float
    memory_updates: List[str]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


class AutonomousCausalExecutor:
    """
    Autonomous system that executes causal meta-goals, processes feedback,
    and continuously improves the memory system through targeted queries.
    """
    
    def __init__(self, causal_generator: CausalMetaGoalGenerator = None,
                 cognitive_system: Phase2AutonomousCognitiveSystem = None):
        self.causal_generator = causal_generator or CausalMetaGoalGenerator()
        self.cognitive_system = cognitive_system or Phase2AutonomousCognitiveSystem()
        
        # Execution state
        self.execution_queue: deque = deque()
        self.execution_history: List[ExecutionResult] = []
        self.is_running = False
        self.execution_thread = None
        
        # Configuration
        self.execution_interval = 30  # 30 seconds between executions
        self.max_concurrent_goals = 3
        self.response_timeout = 60  # 1 minute timeout for responses
        
        # Callback for actually asking questions (to be set by the main system)
        self.question_callback: Optional[Callable[[str], str]] = None
        
        print("[AutonomousCausalExecutor] Initialized autonomous causal execution system")

    def set_question_callback(self, callback: Callable[[str], str]):
        """Set the callback function for asking questions to users."""
        self.question_callback = callback
        print("[AutonomousCausalExecutor] Question callback configured")

    def start_autonomous_execution(self):
        """Start the autonomous execution loop."""
        if self.is_running:
            print("[AutonomousCausalExecutor] Already running")
            return
        
        self.is_running = True
        self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.execution_thread.start()
        print("[AutonomousCausalExecutor] Started autonomous execution loop")

    def stop_autonomous_execution(self):
        """Stop the autonomous execution loop."""
        self.is_running = False
        if self.execution_thread:
            self.execution_thread.join(timeout=5)
        print("[AutonomousCausalExecutor] Stopped autonomous execution loop")

    def _execution_loop(self):
        """Main execution loop that runs autonomously."""
        while self.is_running:
            try:
                # Generate new causal goals if queue is empty
                if len(self.execution_queue) < 2:
                    self._populate_execution_queue()
                
                # Execute next goal if any
                if self.execution_queue:
                    goal_id = self.execution_queue.popleft()
                    self._execute_causal_goal(goal_id)
                
                # Wait before next execution
                time.sleep(self.execution_interval)
                
            except Exception as e:
                print(f"[AutonomousCausalExecutor] Error in execution loop: {e}")
                time.sleep(self.execution_interval)

    def _populate_execution_queue(self):
        """Generate new causal goals and add them to the execution queue."""
        try:
            print("[AutonomousCausalExecutor] Generating new causal goals...")
            
            # Generate causal meta-goals
            goals = self.causal_generator.generate_causal_meta_goals()
            
            # Get high priority goals for execution
            priority_goals = self.causal_generator.get_priority_causal_goals(
                limit=self.max_concurrent_goals
            )
            
            # Add to execution queue
            for goal in priority_goals:
                if goal.goal_id not in self.execution_queue:
                    self.execution_queue.append(goal.goal_id)
                    print(f"[AutonomousCausalExecutor] Queued goal: {goal.description}")
            
        except Exception as e:
            print(f"[AutonomousCausalExecutor] Error populating queue: {e}")

    def _execute_causal_goal(self, goal_id: str) -> ExecutionResult:
        """Execute a specific causal goal."""
        start_time = time.time()
        
        try:
            # Get the goal
            if goal_id not in self.causal_generator.causal_goals:
                return ExecutionResult(
                    goal_id=goal_id,
                    query="",
                    response="",
                    confidence_change=0.0,
                    memory_updates=[],
                    execution_time=time.time() - start_time,
                    success=False,
                    error_message="Goal not found"
                )
            
            goal = self.causal_generator.causal_goals[goal_id]
            
            # Generate query from goal
            queries = self.causal_generator.generate_execution_queries([goal])
            if not queries:
                return ExecutionResult(
                    goal_id=goal_id,
                    query="",
                    response="",
                    confidence_change=0.0,
                    memory_updates=[],
                    execution_time=time.time() - start_time,
                    success=False,
                    error_message="No queries generated"
                )
            
            query = queries[0]
            print(f"[AutonomousCausalExecutor] Executing goal: {goal.description}")
            print(f"[AutonomousCausalExecutor] Query: {query}")
            
            # Execute the query
            if self.question_callback:
                response = self.question_callback(query)
                print(f"[AutonomousCausalExecutor] Response: {response}")
            else:
                # Simulate response for testing
                response = self._simulate_response(goal)
                print(f"[AutonomousCausalExecutor] Simulated response: {response}")
            
            # Process the feedback
            feedback_result = self.causal_generator.process_user_feedback(
                goal_id, response
            )
            
            execution_result = ExecutionResult(
                goal_id=goal_id,
                query=query,
                response=response,
                confidence_change=feedback_result.get("confidence_change", 0.0),
                memory_updates=feedback_result.get("actions_taken", []),
                execution_time=time.time() - start_time,
                success=True
            )
            
            self.execution_history.append(execution_result)
            print(f"[AutonomousCausalExecutor] Successfully executed goal {goal_id}")
            
            return execution_result
            
        except Exception as e:
            execution_result = ExecutionResult(
                goal_id=goal_id,
                query="",
                response="",
                confidence_change=0.0,
                memory_updates=[],
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
            
            self.execution_history.append(execution_result)
            print(f"[AutonomousCausalExecutor] Failed to execute goal {goal_id}: {e}")
            
            return execution_result

    def _simulate_response(self, goal: CausalGoal) -> str:
        """Simulate a response for testing purposes."""
        # Based on goal type, simulate different types of responses
        if goal.improvement_type == "clarification":
            return "Yes, that usually happens when I'm stressed or tired."
        elif goal.improvement_type == "reinforcement":
            return "Yes, that pattern is definitely true for me most of the time."
        elif goal.improvement_type == "gap_filling":
            return "I think it's usually related to my work schedule and sleep quality."
        elif goal.improvement_type == "validation":
            return "Some of those patterns have changed, but others are still accurate."
        else:
            return "I'm not sure, could you be more specific?"

    def execute_specific_goal(self, goal_id: str) -> ExecutionResult:
        """Execute a specific goal manually."""
        return self._execute_causal_goal(goal_id)

    def get_execution_status(self) -> Dict[str, Any]:
        """Get the current status of the execution system."""
        recent_executions = [e for e in self.execution_history 
                           if time.time() - (e.execution_time) < 3600]  # Last hour
        
        successful_executions = [e for e in recent_executions if e.success]
        failed_executions = [e for e in recent_executions if not e.success]
        
        return {
            "is_running": self.is_running,
            "queue_size": len(self.execution_queue),
            "total_executions": len(self.execution_history),
            "recent_executions": len(recent_executions),
            "success_rate": len(successful_executions) / max(1, len(recent_executions)),
            "average_execution_time": sum(e.execution_time for e in recent_executions) / max(1, len(recent_executions)),
            "pending_goals": len([g for g in self.causal_generator.causal_goals.values() if g.status == "pending"]),
            "completed_goals": len([g for g in self.causal_generator.causal_goals.values() if g.status == "completed"])
        }

    def get_recent_results(self, limit: int = 10) -> List[ExecutionResult]:
        """Get recent execution results."""
        return self.execution_history[-limit:] if self.execution_history else []

    def create_manual_intervention(self, user_input: str, context: str = "") -> Dict[str, Any]:
        """
        Create a manual intervention based on user input.
        This allows users to directly influence the causal reasoning.
        """
        print(f"[AutonomousCausalExecutor] Processing manual intervention: {user_input}")
        
        # Process the input through the cognitive system
        result = self.cognitive_system.process_input_with_full_cognition(
            user_input, 
            user_profile_id="manual_intervention",
            session_id=f"intervention_{int(time.time())}"
        )
        
        # Generate new goals based on this intervention
        new_goals = self.causal_generator.generate_causal_meta_goals(
            user_profile_id="manual_intervention"
        )
        
        # Add high priority goals to execution queue
        for goal in new_goals:
            if goal.priority >= 6:  # High priority goals
                self.execution_queue.append(goal.goal_id)
        
        return {
            "processed_input": user_input,
            "facts_extracted": len(result.get("extracted_facts", [])),
            "causal_links_created": result.get("metadata", {}).get("causal_links_created", 0),
            "new_goals_generated": len(new_goals),
            "goals_queued": len([g for g in new_goals if g.priority >= 6])
        }

    def analyze_causal_evolution(self) -> Dict[str, Any]:
        """Analyze how causal understanding has evolved over time."""
        if not self.execution_history:
            return {"status": "no_data"}
        
        # Group executions by goal type
        executions_by_type = {}
        for execution in self.execution_history:
            goal = self.causal_generator.causal_goals.get(execution.goal_id)
            if goal:
                goal_type = goal.improvement_type
                if goal_type not in executions_by_type:
                    executions_by_type[goal_type] = []
                executions_by_type[goal_type].append(execution)
        
        # Calculate improvement metrics
        analysis = {
            "total_executions": len(self.execution_history),
            "execution_types": {
                goal_type: {
                    "count": len(executions),
                    "success_rate": len([e for e in executions if e.success]) / len(executions),
                    "avg_confidence_change": sum(e.confidence_change for e in executions) / len(executions)
                }
                for goal_type, executions in executions_by_type.items()
            },
            "overall_success_rate": len([e for e in self.execution_history if e.success]) / len(self.execution_history),
            "causal_health": self.causal_generator.get_causal_health_summary()
        }
        
        return analysis

    def generate_improvement_report(self) -> List[str]:
        """Generate a report on causal reasoning improvements."""
        analysis = self.analyze_causal_evolution()
        causal_health = analysis.get("causal_health", {})
        
        report = []
        report.append("🧠 CAUSAL REASONING IMPROVEMENT REPORT")
        report.append("=" * 50)
        
        # Overall health
        status = causal_health.get("status", "unknown")
        report.append(f"📊 Overall Status: {status.title()}")
        report.append(f"🔗 Total Causal Facts: {causal_health.get('total_causal_facts', 0)}")
        report.append(f"💪 Average Causal Strength: {causal_health.get('average_causal_strength', 0):.3f}")
        report.append(f"⚡ Strong Links: {causal_health.get('strong_links', 0)}")
        report.append(f"⚠️ Weak Links: {causal_health.get('weak_links', 0)}")
        
        # Execution statistics
        report.append(f"\n🔄 EXECUTION STATISTICS")
        report.append(f"Total Executions: {analysis.get('total_executions', 0)}")
        report.append(f"Success Rate: {analysis.get('overall_success_rate', 0):.1%}")
        
        # Goal progress
        report.append(f"\n🎯 GOAL PROGRESS")
        report.append(f"Pending Goals: {causal_health.get('pending_goals', 0)}")
        report.append(f"Completed Goals: {causal_health.get('completed_goals', 0)}")
        report.append(f"Feedback Sessions: {causal_health.get('feedback_sessions', 0)}")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS")
        if causal_health.get('weak_links', 0) > causal_health.get('strong_links', 0):
            report.append("• Focus on clarifying weak causal connections")
        if causal_health.get('pending_goals', 0) > 5:
            report.append("• Increase execution frequency to process pending goals")
        if analysis.get('overall_success_rate', 0) < 0.7:
            report.append("• Review and improve question generation strategies")
        
        return report 