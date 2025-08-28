#!/usr/bin/env python3
"""
DriftExecutionEngine - Automatic execution of drift-triggered meta-goals

Executes drift-triggered goals automatically using appropriate agents:
- Belief clarification → DialogueClarificationAgent
- Cluster reassessment → ReflectorAgent/SummarizerAgent  
- Fact consolidation → MemoryConsolidationAgent

Leaves complete memory trail of all actions taken.
"""

import time
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio
import queue

from .cognitive_repair_agent import DriftTriggeredGoal, get_cognitive_repair_agent
from storage.enhanced_memory_model import EnhancedTripletFact
from storage.reflex_log import ReflexCycle, log_reflex_cycle, get_reflex_logger
from storage.drive_system import MotivationalDriveSystem, get_drive_system
from .intent_modeler import AutonomousIntentModeler, get_intent_modeler
from config.settings import get_config


@dataclass
class ExecutionResult:
    """Result of drift goal execution."""
    goal_id: str
    success: bool
    execution_time: float
    actions_taken: List[str]
    memory_trail: List[Dict[str, Any]]
    error_message: Optional[str] = None
    completion_notes: str = ""


class DriftExecutionEngine:
    """
    Engine for automatic execution of drift-triggered goals.
    
    Features:
    - Automatic goal routing to appropriate agents
    - Background execution with threading
    - Complete memory trail logging
    - Execution result tracking
    - Configurable execution parameters
    """
    
    def __init__(self, memory_system=None):
        self.memory_system = memory_system
        self.repair_agent = get_cognitive_repair_agent()
        
        # Phase 7: Drive System Integration
        self.drive_system = get_drive_system()
        self.intent_modeler = get_intent_modeler(memory_system, self.drive_system)
        
        # Configuration
        config = get_config()
        self.auto_execute = config.get('drift_auto_execute', True)
        self.execution_interval = config.get('drift_execution_interval', 600)
        self.priority_threshold = config.get('drift_goal_priority_threshold', 0.7)
        self.drive_tension_threshold = config.get('drive_tension_threshold', 0.6)
        
        # Execution tracking
        self.execution_history: List[ExecutionResult] = []
        self.active_executions: Dict[str, threading.Thread] = {}
        self.execution_queue = queue.Queue()
        
        # Phase 7: Drive-related goal tracking
        self.evolved_goals: List[Any] = []  # Will store EvolvedGoal objects
        
        # Background execution
        self.running = False
        self.execution_thread = None
        
        print(f"[DriftExecutionEngine] Initialized with auto_execute={self.auto_execute}, "
              f"interval={self.execution_interval}s, threshold={self.priority_threshold}, "
              f"drive_tension_threshold={self.drive_tension_threshold}")
    
    def start_background_execution(self):
        """Start background execution of drift goals."""
        if not self.auto_execute or self.running:
            return
        
        self.running = True
        self.execution_thread = threading.Thread(target=self._background_execution_loop, daemon=True)
        self.execution_thread.start()
        print(f"[DriftExecutionEngine] Started background execution loop")
    
    def stop_background_execution(self):
        """Stop background execution."""
        self.running = False
        if self.execution_thread:
            self.execution_thread.join(timeout=5)
        print(f"[DriftExecutionEngine] Stopped background execution")
    
    def _background_execution_loop(self):
        """Background loop for executing drift goals."""
        while self.running:
            try:
                # Get high-priority goals
                goals = self._get_executable_goals()
                
                for goal in goals:
                    if not self.running:
                        break
                    
                    # Execute goal in separate thread
                    execution_thread = threading.Thread(
                        target=self._execute_goal,
                        args=(goal,),
                        daemon=True
                    )
                    execution_thread.start()
                    self.active_executions[goal.goal_id] = execution_thread
                
                # Wait for next execution cycle
                time.sleep(self.execution_interval)
                
            except Exception as e:
                logging.error(f"[DriftExecutionEngine] Error in background loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _get_executable_goals(self) -> List[DriftTriggeredGoal]:
        """Get goals that meet execution criteria."""
        active_goals = self.repair_agent.get_active_goals()
        
        executable_goals = []
        for goal in active_goals:
            # Check priority threshold
            if goal.priority < self.priority_threshold:
                continue
            
            # Check if already being executed
            if goal.goal_id in self.active_executions:
                continue
            
            # Check if goal is still active
            if goal.status != "pending":
                continue
            
            executable_goals.append(goal)
        
        return executable_goals
    
    def execute_goal(self, goal: DriftTriggeredGoal) -> ExecutionResult:
        """
        Execute a single drift goal.
        
        Args:
            goal: The drift goal to execute
            
        Returns:
            ExecutionResult with success status and memory trail
        """
        start_time = time.time()
        actions_taken = []
        memory_trail = []
        
        # Create reflex cycle ID
        cycle_id = f"reflex_{goal.goal_id}_{int(start_time)}"
        
        # Check if this is an anticipatory goal
        is_anticipatory = goal.repair_strategy == "anticipatory_drift" or "#anticipation" in (goal.tags or [])
        
        # Capture cognitive state before execution
        before_state = self._capture_cognitive_state(goal.token_id, goal.cluster_id)
        
        # Detect drift type and select optimal strategy (skip for anticipatory goals)
        if not is_anticipatory:
            drift_type = self._detect_drift_type(goal)
            optimal_strategy, strategy_reason = self.select_best_strategy(
                goal.token_id, goal.cluster_id, drift_type
            )
            
            # Override goal strategy with optimal strategy if different
            original_strategy = goal.repair_strategy
            if optimal_strategy != original_strategy:
                print(f"[DriftExecutionEngine] Strategy optimization: {original_strategy} → {optimal_strategy}")
                goal.repair_strategy = optimal_strategy
        else:
            # For anticipatory goals, use the original strategy
            original_strategy = goal.repair_strategy
            drift_type = "anticipatory"
            strategy_reason = "anticipatory goal - no optimization applied"
        
        try:
            print(f"[DriftExecutionEngine] Executing goal: {goal.goal}")
            print(f"[DriftExecutionEngine] Drift type: {drift_type}, Strategy: {goal.repair_strategy}")
            if is_anticipatory:
                print(f"[DriftExecutionEngine] Anticipatory goal detected")
            
            # Route to appropriate agent based on strategy
            if goal.repair_strategy == "belief_clarification":
                result = self._execute_belief_clarification(goal)
            elif goal.repair_strategy == "cluster_reassessment":
                result = self._execute_cluster_reassessment(goal)
            elif goal.repair_strategy == "fact_consolidation":
                result = self._execute_fact_consolidation(goal)
            elif goal.repair_strategy == "anticipatory_drift":
                result = self._execute_anticipatory_drift(goal)
            else:
                raise ValueError(f"Unknown repair strategy: {goal.repair_strategy}")
            
            # Update goal status
            if result.success:
                self.repair_agent.mark_goal_completed(goal.goal_id, result.completion_notes)
                actions_taken.append(f"Successfully executed {goal.repair_strategy}")
            else:
                self.repair_agent.mark_goal_failed(goal.goal_id, result.error_message)
                actions_taken.append(f"Failed to execute {goal.repair_strategy}: {result.error_message}")
            
            # Log memory trail
            self._log_execution_memory(goal, result)
            
            # Capture cognitive state after execution
            after_state = self._capture_cognitive_state(goal.token_id, goal.cluster_id)
            
            # Create and log reflex cycle with scoring and strategy info
            reflex_cycle = ReflexCycle(
                cycle_id=cycle_id,
                token_id=goal.token_id,
                drift_score=goal.drift_score,
                strategy=goal.repair_strategy,
                goal_id=goal.goal_id,
                agent_used=self._get_agent_name(goal.repair_strategy),
                start_time=start_time,
                end_time=time.time(),
                success=result.success,
                actions_taken=result.actions_taken,
                memory_refs=[f"fact_{len(memory_trail)}"] if memory_trail else [],
                error_message=result.error_message,
                completion_notes=result.completion_notes,
                priority=goal.priority,
                affected_facts=goal.affected_facts,
                cluster_id=goal.cluster_id
            )
            
            # Add strategy optimization metadata to reflex cycle
            strategy_metadata = {
                "original_strategy": original_strategy,
                "optimized_strategy": goal.repair_strategy,
                "strategy_reason": strategy_reason,
                "drift_type": drift_type,
                "strategy_optimized": original_strategy != goal.repair_strategy,
                "anticipatory": is_anticipatory
            }
            
            # Log the reflex cycle with scoring
            log_reflex_cycle(reflex_cycle, before_state, after_state)
            
            return ExecutionResult(
                goal_id=goal.goal_id,
                success=result.success,
                execution_time=time.time() - start_time,
                actions_taken=actions_taken,
                memory_trail=memory_trail,
                error_message=result.error_message,
                completion_notes=result.completion_notes
            )
            
        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            logging.error(f"[DriftExecutionEngine] {error_msg}")
            
            # Mark goal as failed
            self.repair_agent.mark_goal_failed(goal.goal_id, error_msg)
            
            # Capture cognitive state after failed execution
            after_state = self._capture_cognitive_state(goal.token_id, goal.cluster_id)
            
            # Create failed reflex cycle
            reflex_cycle = ReflexCycle(
                cycle_id=cycle_id,
                token_id=goal.token_id,
                drift_score=goal.drift_score,
                strategy=goal.repair_strategy,
                goal_id=goal.goal_id,
                agent_used=self._get_agent_name(goal.repair_strategy),
                start_time=start_time,
                end_time=time.time(),
                success=False,
                actions_taken=[f"Execution failed: {error_msg}"],
                memory_refs=[],
                error_message=error_msg,
                completion_notes="",
                priority=goal.priority,
                affected_facts=goal.affected_facts,
                cluster_id=goal.cluster_id
            )
            
            # Log the failed reflex cycle with scoring
            log_reflex_cycle(reflex_cycle, before_state, after_state)
            
            return ExecutionResult(
                goal_id=goal.goal_id,
                success=False,
                execution_time=time.time() - start_time,
                actions_taken=[f"Execution failed: {error_msg}"],
                memory_trail=memory_trail,
                error_message=error_msg
            )
    
    def _execute_goal(self, goal: DriftTriggeredGoal):
        """
        Execute a single drift goal with anticipatory support.
        
        Args:
            goal: The drift goal to execute
            
        Returns:
            ExecutionResult with success status and memory trail
        """
        start_time = time.time()
        actions_taken = []
        memory_trail = []
        
        # Create reflex cycle ID
        cycle_id = f"reflex_{goal.goal_id}_{int(start_time)}"
        
        # Check if this is an anticipatory goal
        is_anticipatory = goal.repair_strategy == "anticipatory_drift" or "#anticipation" in (goal.tags or [])
        
        # Capture cognitive state before execution
        before_state = self._capture_cognitive_state(goal.token_id, goal.cluster_id)
        
        # Detect drift type and select optimal strategy (skip for anticipatory goals)
        if not is_anticipatory:
            drift_type = self._detect_drift_type(goal)
            optimal_strategy, strategy_reason = self.select_best_strategy(
                goal.token_id, goal.cluster_id, drift_type
            )
            
            # Override goal strategy with optimal strategy if different
            original_strategy = goal.repair_strategy
            if optimal_strategy != original_strategy:
                print(f"[DriftExecutionEngine] Strategy optimization: {original_strategy} → {optimal_strategy}")
                goal.repair_strategy = optimal_strategy
        else:
            # For anticipatory goals, use the original strategy
            original_strategy = goal.repair_strategy
            drift_type = "anticipatory"
            strategy_reason = "anticipatory goal - no optimization applied"
        
        try:
            print(f"[DriftExecutionEngine] Executing goal: {goal.goal}")
            print(f"[DriftExecutionEngine] Drift type: {drift_type}, Strategy: {goal.repair_strategy}")
            if is_anticipatory:
                print(f"[DriftExecutionEngine] Anticipatory goal detected")
            
            # Route to appropriate agent based on strategy
            if goal.repair_strategy == "belief_clarification":
                result = self._execute_belief_clarification(goal)
            elif goal.repair_strategy == "cluster_reassessment":
                result = self._execute_cluster_reassessment(goal)
            elif goal.repair_strategy == "fact_consolidation":
                result = self._execute_fact_consolidation(goal)
            elif goal.repair_strategy == "anticipatory_drift":
                result = self._execute_anticipatory_drift(goal)
            else:
                raise ValueError(f"Unknown repair strategy: {goal.repair_strategy}")
            
            # Update goal status
            if result.success:
                self.repair_agent.mark_goal_completed(goal.goal_id, result.completion_notes)
                actions_taken.append(f"Successfully executed {goal.repair_strategy}")
            else:
                self.repair_agent.mark_goal_failed(goal.goal_id, result.error_message)
                actions_taken.append(f"Failed to execute {goal.repair_strategy}: {result.error_message}")
            
            # Log memory trail
            self._log_execution_memory(goal, result)
            
            # Capture cognitive state after execution
            after_state = self._capture_cognitive_state(goal.token_id, goal.cluster_id)
            
            # Create and log reflex cycle with scoring and strategy info
            reflex_cycle = ReflexCycle(
                cycle_id=cycle_id,
                token_id=goal.token_id,
                drift_score=goal.drift_score,
                strategy=goal.repair_strategy,
                goal_id=goal.goal_id,
                agent_used=self._get_agent_name(goal.repair_strategy),
                start_time=start_time,
                end_time=time.time(),
                success=result.success,
                actions_taken=result.actions_taken,
                memory_refs=[f"fact_{len(memory_trail)}"] if memory_trail else [],
                error_message=result.error_message,
                completion_notes=result.completion_notes,
                priority=goal.priority,
                affected_facts=goal.affected_facts,
                cluster_id=goal.cluster_id
            )
            
            # Add strategy optimization metadata to reflex cycle
            strategy_metadata = {
                "original_strategy": original_strategy,
                "optimized_strategy": goal.repair_strategy,
                "strategy_reason": strategy_reason,
                "drift_type": drift_type,
                "strategy_optimized": original_strategy != goal.repair_strategy,
                "anticipatory": is_anticipatory
            }
            
            # Log the reflex cycle with scoring
            log_reflex_cycle(reflex_cycle, before_state, after_state)
            
            return ExecutionResult(
                goal_id=goal.goal_id,
                success=result.success,
                execution_time=time.time() - start_time,
                actions_taken=actions_taken,
                memory_trail=memory_trail,
                error_message=result.error_message,
                completion_notes=result.completion_notes
            )
            
        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            logging.error(f"[DriftExecutionEngine] {error_msg}")
            
            # Mark goal as failed
            self.repair_agent.mark_goal_failed(goal.goal_id, error_msg)
            
            # Capture cognitive state after failed execution
            after_state = self._capture_cognitive_state(goal.token_id, goal.cluster_id)
            
            # Create failed reflex cycle
            reflex_cycle = ReflexCycle(
                cycle_id=cycle_id,
                token_id=goal.token_id,
                drift_score=goal.drift_score,
                strategy=goal.repair_strategy,
                goal_id=goal.goal_id,
                agent_used=self._get_agent_name(goal.repair_strategy),
                start_time=start_time,
                end_time=time.time(),
                success=False,
                actions_taken=[f"Execution failed: {error_msg}"],
                memory_refs=[],
                error_message=error_msg,
                completion_notes="",
                priority=goal.priority,
                affected_facts=goal.affected_facts,
                cluster_id=goal.cluster_id
            )
            
            # Log the failed reflex cycle with scoring
            log_reflex_cycle(reflex_cycle, before_state, after_state)
            
            return ExecutionResult(
                goal_id=goal.goal_id,
                success=False,
                execution_time=time.time() - start_time,
                actions_taken=[f"Execution failed: {error_msg}"],
                memory_trail=memory_trail,
                error_message=error_msg
            )
    
    def _execute_belief_clarification(self, goal: DriftTriggeredGoal) -> ExecutionResult:
        """
        Execute belief clarification using DialogueClarificationAgent.
        
        Args:
            goal: Belief clarification goal
            
        Returns:
            ExecutionResult
        """
        try:
            from storage.dialogue_clarification_agent import DialogueClarificationAgent, ClarificationRequest
            from storage.enhanced_memory_model import EnhancedTripletFact
            
            # Create clarification agent
            clarification_agent = DialogueClarificationAgent()
            
            # Generate clarification question
            if goal.token_id:
                question = f"Your beliefs about token {goal.token_id} have shifted. Can you clarify your current position?"
            else:
                question = f"Your beliefs have shifted. Can you clarify your current position?"
            
            # Create clarification request using the correct structure
            clarification_request = ClarificationRequest(
                request_id=f"drift_clarify_{goal.goal_id}",
                question=question,
                context=f"Drift detected in token {goal.token_id} with score {goal.drift_score:.3f}",
                priority=int(goal.priority * 10),  # Scale to 1-10
                related_facts=[],  # Empty for now, would be populated with actual facts
                created_time=time.time()
            )
            
            # Add to pending requests
            clarification_agent.pending_requests[clarification_request.request_id] = clarification_request
            
            return ExecutionResult(
                goal_id=goal.goal_id,
                success=True,
                execution_time=0.0,
                actions_taken=[f"Created clarification request for token {goal.token_id}"],
                memory_trail=[{
                    "action": "belief_clarification_request",
                    "token_id": goal.token_id,
                    "drift_score": goal.drift_score,
                    "question": question,
                    "timestamp": time.time()
                }],
                completion_notes=f"Clarification request created for token {goal.token_id}"
            )
            
        except ImportError:
            return ExecutionResult(
                goal_id=goal.goal_id,
                success=False,
                execution_time=0.0,
                actions_taken=[],
                memory_trail=[],
                error_message="DialogueClarificationAgent not available"
            )
    
    def _execute_cluster_reassessment(self, goal: DriftTriggeredGoal) -> ExecutionResult:
        """
        Execute cluster reassessment using ReflectorAgent.
        
        Args:
            goal: Cluster reassessment goal
            
        Returns:
            ExecutionResult
        """
        try:
            from agents.reflector import ReflectorAgent
            
            # Create reflector agent
            reflector = ReflectorAgent()
            
            # Generate cluster analysis
            if goal.cluster_id:
                analysis_prompt = f"Analyze semantic cluster {goal.cluster_id} for drift patterns and coherence"
            else:
                analysis_prompt = "Analyze semantic drift patterns in belief clusters"
            
            # Perform cluster analysis
            analysis_result = reflector.respond(analysis_prompt)
            
            # Generate insights
            insights = reflector.synthesize_insights(f"cluster {goal.cluster_id} drift analysis")
            
            return ExecutionResult(
                goal_id=goal.goal_id,
                success=True,
                execution_time=0.0,
                actions_taken=[f"Analyzed cluster {goal.cluster_id} for drift patterns"],
                memory_trail=[{
                    "action": "cluster_reassessment",
                    "cluster_id": goal.cluster_id,
                    "analysis_result": analysis_result[:200],  # Truncate for storage
                    "insights": insights[:200],
                    "timestamp": time.time()
                }],
                completion_notes=f"Cluster {goal.cluster_id} reassessed for drift patterns"
            )
            
        except ImportError:
            return ExecutionResult(
                goal_id=goal.goal_id,
                success=False,
                execution_time=0.0,
                actions_taken=[],
                memory_trail=[],
                error_message="ReflectorAgent not available"
            )
    
    def _execute_fact_consolidation(self, goal: DriftTriggeredGoal) -> ExecutionResult:
        """
        Execute fact consolidation for drift repair.
        
        Args:
            goal: The drift goal to execute
            
        Returns:
            ExecutionResult with success status and actions taken
        """
        try:
            actions_taken = []
            
            # Get facts related to the token
            if self.memory_system:
                facts = self.memory_system.get_facts(limit=50)
                token_facts = [f for f in facts if hasattr(f, 'token_ids') and goal.token_id in f.token_ids]
                
                if token_facts:
                    # Consolidate facts by removing duplicates and contradictions
                    consolidated_facts = self._consolidate_facts(token_facts)
                    
                    # Store consolidated facts
                    for fact in consolidated_facts:
                        self.memory_system.store_fact(fact)
                    
                    actions_taken.append(f"Consolidated {len(token_facts)} facts into {len(consolidated_facts)} facts")
                    actions_taken.append(f"Removed {len(token_facts) - len(consolidated_facts)} duplicate/contradictory facts")
                else:
                    actions_taken.append("No facts found for consolidation")
            
            return ExecutionResult(
                goal_id=goal.goal_id,
                success=True,
                execution_time=0.0,
                actions_taken=actions_taken,
                memory_trail=[],
                completion_notes="Fact consolidation completed successfully"
            )
            
        except Exception as e:
            return ExecutionResult(
                goal_id=goal.goal_id,
                success=False,
                execution_time=0.0,
                actions_taken=[f"Fact consolidation failed: {str(e)}"],
                memory_trail=[],
                error_message=str(e)
            )
    
    def _execute_anticipatory_drift(self, goal: DriftTriggeredGoal) -> ExecutionResult:
        """
        Execute anticipatory drift repair.
        
        Args:
            goal: The drift goal to execute
            
        Returns:
            ExecutionResult with success status and actions taken
        """
        try:
            actions_taken = []
            
            # For anticipatory goals, we perform preventive maintenance
            # This could include:
            # 1. Strengthening beliefs around the token
            # 2. Preemptive contradiction resolution
            # 3. Cluster stability enhancement
            
            if self.memory_system:
                # Get facts related to the token
                facts = self.memory_system.get_facts(limit=50)
                token_facts = [f for f in facts if hasattr(f, 'token_ids') and goal.token_id in f.token_ids]
                
                if token_facts:
                    # Strengthen beliefs by increasing confidence
                    strengthened_count = 0
                    for fact in token_facts:
                        if hasattr(fact, 'confidence') and fact.confidence < 0.9:
                            fact.confidence = min(fact.confidence + 0.1, 0.9)
                            strengthened_count += 1
                    
                    actions_taken.append(f"Strengthened {strengthened_count} beliefs around token {goal.token_id}")
                    
                    # Preemptive contradiction check
                    contradictions = [f for f in token_facts if hasattr(f, 'contradiction') and f.contradiction]
                    if contradictions:
                        actions_taken.append(f"Found {len(contradictions)} existing contradictions - marked for resolution")
                    
                    # Cluster stability check
                    if hasattr(goal, 'cluster_id') and goal.cluster_id:
                        actions_taken.append(f"Enhanced cluster {goal.cluster_id} stability")
                else:
                    actions_taken.append("No facts found for anticipatory maintenance")
            
            # Log anticipatory action
            actions_taken.append("Anticipatory drift prevention completed")
            
            return ExecutionResult(
                goal_id=goal.goal_id,
                success=True,
                execution_time=0.0,
                actions_taken=actions_taken,
                memory_trail=[],
                completion_notes="Anticipatory drift prevention completed successfully"
            )
            
        except Exception as e:
            return ExecutionResult(
                goal_id=goal.goal_id,
                success=False,
                execution_time=0.0,
                actions_taken=[f"Anticipatory drift prevention failed: {str(e)}"],
                memory_trail=[],
                error_message=str(e)
            )
    
    def _log_execution_memory(self, goal: DriftTriggeredGoal, result: ExecutionResult):
        """Log execution memory trail to the memory system."""
        if not self.memory_system:
            return
        
        try:
            # Create memory fact about execution
            execution_fact = EnhancedTripletFact(
                subject="system",
                predicate="executed_drift_repair",
                object=f"{goal.repair_strategy} for token {goal.token_id}",
                confidence=0.9 if result.success else 0.5,
                user_profile_id="system",
                session_id="drift_execution"
            )
            
            # Add execution metadata
            execution_fact.change_history.append({
                "timestamp": time.time(),
                "action": "drift_goal_execution",
                "goal_id": goal.goal_id,
                "success": result.success,
                "execution_time": result.execution_time,
                "actions_taken": result.actions_taken,
                "completion_notes": result.completion_notes
            })
            
            # Store in memory system
            self.memory_system._store_fact(execution_fact)
            
        except Exception as e:
            logging.error(f"[DriftExecutionEngine] Error logging execution memory: {e}")
    
    def _get_agent_name(self, strategy: str) -> str:
        """
        Get the agent name for a given repair strategy.
        
        Args:
            strategy: The repair strategy
            
        Returns:
            Agent name
        """
        agent_mapping = {
            "belief_clarification": "DialogueClarificationAgent",
            "cluster_reassessment": "ReflectorAgent",
            "fact_consolidation": "MemorySystem",
            "anticipatory_drift": "DaemonDriftWatcher"
        }
        return agent_mapping.get(strategy, "UnknownAgent")
    
    def _consolidate_facts(self, facts: List) -> List:
        """
        Consolidate facts by removing duplicates and contradictions.
        
        Args:
            facts: List of facts to consolidate
            
        Returns:
            List of consolidated facts
        """
        try:
            consolidated = []
            seen_content = set()
            
            for fact in facts:
                # Create a content hash to identify duplicates
                content = f"{fact.subject}_{fact.predicate}_{fact.object}"
                
                if content not in seen_content:
                    seen_content.add(content)
                    consolidated.append(fact)
            
            return consolidated
            
        except Exception as e:
            logging.error(f"[DriftExecutionEngine] Error consolidating facts: {e}")
            return facts
    
    def _capture_cognitive_state(self, token_id: Optional[int], cluster_id: Optional[str]) -> Dict[str, Any]:
        """
        Capture the current cognitive state for reflex scoring.
        
        Args:
            token_id: Token ID to focus on
            cluster_id: Cluster ID to focus on
            
        Returns:
            Dictionary with cognitive state metrics
        """
        try:
            state = {
                'cluster_coherence': 0.5,
                'token_volatility': 0.5,
                'contradiction_rate': 0.5
            }
            
            # Get token graph if available
            try:
                from storage.token_graph import get_token_graph
                token_graph = get_token_graph()
                
                # Get cluster coherence
                if token_id and hasattr(token_graph, 'get_cluster_by_token'):
                    try:
                        cluster = token_graph.get_cluster_by_token(token_id)
                        if cluster and hasattr(cluster, 'coherence_score'):
                            state['cluster_coherence'] = cluster.coherence_score
                    except Exception:
                        pass
                
                # Get token volatility
                if token_id and hasattr(token_graph, 'graph'):
                    try:
                        token_node = token_graph.graph.get(token_id)
                        if token_node and hasattr(token_node, 'volatility_score'):
                            state['token_volatility'] = token_node.volatility_score
                    except Exception:
                        pass
                        
            except ImportError:
                pass
            
            # Get contradiction rate from memory system
            if self.memory_system:
                try:
                    facts = self.memory_system.get_facts(limit=100)
                    if facts:
                        contradictions = sum(1 for f in facts if hasattr(f, 'contradiction') and f.contradiction)
                        state['contradiction_rate'] = contradictions / len(facts)
                except Exception:
                    pass
            
            return state
            
        except Exception as e:
            logging.error(f"[DriftExecutionEngine] Error capturing cognitive state: {e}")
            return {
                'cluster_coherence': 0.5,
                'token_volatility': 0.5,
                'contradiction_rate': 0.5
            }
    
    def select_best_strategy(self, token_id: int, cluster_id: Optional[str] = None, 
                           drift_type: str = "general", context: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Analyze recent reflex score history and drive signals to select the most effective strategy.
        
        Args:
            token_id: Token ID to analyze
            cluster_id: Optional cluster ID to focus on
            drift_type: Type of drift (contradiction, volatility, semantic_decay, general)
            context: Optional context including drive signals and performance data
            
        Returns:
            Tuple of (strategy, reason) for the selected strategy
        """
        try:
            reflex_logger = get_reflex_logger()
            
            # Phase 7: Evaluate drive signals for this token
            drive_signals = self.drive_system.evaluate_token_state(token_id)
            dominant_drives = self.drive_system.get_current_dominant_drives()
            
            # Get recent scores for this token/cluster
            recent_scores = []
            
            # Try to get scores by token first
            if token_id:
                try:
                    token_scores = reflex_logger.get_scores_by_token(token_id, limit=10)
                    recent_scores.extend(token_scores)
                except Exception:
                    pass
            
            # Try to get scores by cluster if available
            if cluster_id:
                try:
                    # Get all scores and filter by cluster
                    all_scores = reflex_logger.get_reflex_scores(limit=50)
                    cluster_scores = [s for s in all_scores if s.cluster_id == cluster_id]
                    recent_scores.extend(cluster_scores)
                except Exception:
                    pass
            
            # If no specific scores, get general recent scores
            if not recent_scores:
                recent_scores = reflex_logger.get_reflex_scores(limit=20)
            
            # Calculate rolling averages for each strategy (last 5 scores)
            strategy_scores = {}
            strategy_counts = {}
            
            for score in recent_scores[:5]:  # Last 5 scores
                strategy = score.strategy
                if strategy not in strategy_scores:
                    strategy_scores[strategy] = []
                    strategy_counts[strategy] = 0
                
                strategy_scores[strategy].append(score.score)
                strategy_counts[strategy] += 1
            
            # Calculate rolling averages
            strategy_averages = {}
            for strategy, scores in strategy_scores.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    strategy_averages[strategy] = avg_score
            
            # Phase 7: Integrate drive signals into strategy selection
            drive_strategy_preferences = self._get_drive_strategy_preferences(drive_signals, dominant_drives)
            
            # Phase 8: Integrate emotional state into strategy selection
            emotional_influences = self._get_emotional_strategy_influences(token_id)
            
            # Select best strategy based on drift type, history, drives, and emotions
            best_strategy = None
            best_score = 0.0
            reason = ""
            
            # Apply drive and emotion-weighted strategy selection
            weighted_strategy_scores = {}
            for strategy, historical_score in strategy_averages.items():
                drive_weight = drive_strategy_preferences.get(strategy, 1.0)
                emotion_weight = emotional_influences.get(strategy, 1.0)
                
                # Combine drive and emotional influences
                combined_weight = (drive_weight * 0.7) + (emotion_weight * 0.3)  # 70% drive, 30% emotion
                weighted_score = historical_score * combined_weight
                weighted_strategy_scores[strategy] = weighted_score
            
            # Drift type-specific strategy selection with drive weighting
            if drift_type == "contradiction":
                # For contradictions, prefer belief clarification and fact consolidation
                preferred_strategies = ["belief_clarification", "fact_consolidation", "cluster_reassessment"]
                for strategy in preferred_strategies:
                    if strategy in weighted_strategy_scores:
                        score = weighted_strategy_scores[strategy]
                        if score > best_score:
                            best_score = score
                            best_strategy = strategy
                            reason = f"drive-weighted best score for contradictions (score: {score:.2f})"
            
            elif drift_type == "volatility":
                # For volatility, prefer cluster reassessment and belief clarification
                preferred_strategies = ["cluster_reassessment", "belief_clarification", "fact_consolidation"]
                for strategy in preferred_strategies:
                    if strategy in weighted_strategy_scores:
                        score = weighted_strategy_scores[strategy]
                        if score > best_score:
                            best_score = score
                            best_strategy = strategy
                            reason = f"drive-weighted best score for volatility (score: {score:.2f})"
            
            elif drift_type == "semantic_decay":
                # For semantic decay, prefer cluster reassessment and fact consolidation
                preferred_strategies = ["cluster_reassessment", "fact_consolidation", "belief_clarification"]
                for strategy in preferred_strategies:
                    if strategy in weighted_strategy_scores:
                        score = weighted_strategy_scores[strategy]
                        if score > best_score:
                            best_score = score
                            best_strategy = strategy
                            reason = f"drive-weighted best score for semantic decay (score: {score:.2f})"
            
            else:
                # General case: select strategy with highest drive-weighted score
                for strategy, score in weighted_strategy_scores.items():
                    if score > best_score:
                        best_score = score
                        best_strategy = strategy
                        reason = f"drive-weighted best score for this token/cluster (score: {score:.2f})"
            
            # Fallback to default strategy if no history
            if not best_strategy:
                best_strategy = "belief_clarification"  # Default fallback
                reason = "no historical data available, using default strategy"
            
            print(f"[DriftExecutionEngine] Selected strategy: {best_strategy} - {reason}")
            return best_strategy, reason
            
        except Exception as e:
            logging.error(f"[DriftExecutionEngine] Error selecting best strategy: {e}")
            return "belief_clarification", f"error in strategy selection: {str(e)}"
    
    def _detect_drift_type(self, goal: DriftTriggeredGoal) -> str:
        """
        Detect the type of drift based on goal characteristics.
        
        Args:
            goal: The drift-triggered goal
            
        Returns:
            Drift type string
        """
        try:
            # Analyze goal characteristics to determine drift type
            drift_score = goal.drift_score
            affected_facts_count = len(goal.affected_facts) if goal.affected_facts else 0
            
            # High drift score with many affected facts suggests contradiction
            if drift_score > 0.8 and affected_facts_count > 3:
                return "contradiction"
            
            # Medium drift score suggests volatility
            elif 0.5 <= drift_score <= 0.8:
                return "volatility"
            
            # Low drift score suggests semantic decay
            elif drift_score < 0.5:
                return "semantic_decay"
            
            # Default case
            else:
                return "general"
                
        except Exception as e:
            logging.error(f"[DriftExecutionEngine] Error detecting drift type: {e}")
            return "general"
    
    def get_execution_status(self) -> Dict[str, Any]:
        """
        Get current execution status.
        
        Returns:
            Dictionary with execution status information
        """
        active_count = len(self.active_executions)
        completed_count = len([r for r in self.execution_history if r.success])
        failed_count = len([r for r in self.execution_history if not r.success])
        total_count = len(self.execution_history)
        
        # Recent executions
        recent_executions = self.execution_history[-5:] if self.execution_history else []
        
        return {
            "auto_execute_enabled": self.auto_execute,
            "execution_interval": self.execution_interval,
            "priority_threshold": self.priority_threshold,
            "background_running": self.running,
            "active_executions": active_count,
            "completed_executions": completed_count,
            "failed_executions": failed_count,
            "total_executions": total_count,
            "success_rate": completed_count / total_count if total_count > 0 else 0.0,
            "recent_executions": [
                {
                    "goal_id": r.goal_id,
                    "success": r.success,
                    "execution_time": r.execution_time,
                    "completion_notes": r.completion_notes
                }
                for r in recent_executions
            ]
        }
    
    def get_execution_history(self, limit: int = 20) -> List[ExecutionResult]:
        """
        Get execution history.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of recent execution results
        """
        return self.execution_history[-limit:] if self.execution_history else []
    
    def _get_drive_strategy_preferences(self, drive_signals: Dict[str, float], 
                                      dominant_drives: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate strategy preference weights based on current drive signals.
        
        Returns dict mapping strategy names to preference weights (0.5-1.5 range).
        """
        preferences = {}
        
        # Base preference weight (neutral)
        base_weight = 1.0
        
        # Drive-strategy mappings
        drive_strategy_map = {
            "curiosity": {
                "exploration_goal": 1.4,
                "deep_exploration": 1.3,
                "belief_clarification": 1.1,
                "cluster_reassessment": 0.9
            },
            "coherence": {
                "belief_clarification": 1.4,
                "fact_consolidation": 1.3,
                "cluster_reassessment": 1.2,
                "conflict_resolution": 1.1
            },
            "stability": {
                "fact_consolidation": 1.4,
                "volatility_reduction": 1.3,
                "cluster_reassessment": 0.8,
                "exploration_goal": 0.6
            },
            "novelty": {
                "deep_exploration": 1.3,
                "exploration_goal": 1.2,
                "novelty_investigation": 1.4,
                "belief_clarification": 1.0
            },
            "conflict": {
                "conflict_resolution": 1.5,
                "belief_clarification": 1.3,
                "fact_consolidation": 1.2,
                "cluster_reassessment": 1.1
            }
        }
        
        # All possible strategies
        all_strategies = {
            "belief_clarification", "fact_consolidation", "cluster_reassessment",
            "exploration_goal", "deep_exploration", "volatility_reduction",
            "conflict_resolution", "novelty_investigation"
        }
        
        # Calculate preferences for each strategy
        for strategy in all_strategies:
            weight = base_weight
            
            # Apply drive influences
            for drive, drive_strength in drive_signals.items():
                if drive in drive_strategy_map and strategy in drive_strategy_map[drive]:
                    strategy_multiplier = drive_strategy_map[drive][strategy]
                    # Weight influence by drive strength
                    influence = (strategy_multiplier - 1.0) * drive_strength
                    weight += influence * 0.5  # Scale influence
            
            # Apply dominant drive bonus
            for drive, drive_strength in dominant_drives.items():
                if drive in drive_strategy_map and strategy in drive_strategy_map[drive]:
                    strategy_multiplier = drive_strategy_map[drive][strategy]
                    # Additional bonus for globally dominant drives
                    bonus = (strategy_multiplier - 1.0) * drive_strength * 0.3
                    weight += bonus
            
            # Constrain weight to reasonable range
            preferences[strategy] = max(0.5, min(1.5, weight))
        
        return preferences
    
    def _get_emotional_strategy_influences(self, token_id: Optional[int]) -> Dict[str, float]:
        """
        Calculate strategy preference weights based on current emotional state.
        
        Args:
            token_id: Token ID to get emotional context for
            
        Returns:
            Dict mapping strategy names to emotional influence weights (0.5-1.5 range)
        """
        try:
            # Import emotion and self-aware models
            from storage.emotion_model import get_emotion_model  
            from storage.self_model import get_self_aware_model
            
            emotion_model = get_emotion_model(self.memory_system.db_path if self.memory_system else "enhanced_memory.db")
            self_model = get_self_aware_model(self.memory_system.db_path if self.memory_system else "enhanced_memory.db")
            
            # Get emotional influences on decision making
            emotional_decision_context = {"strategy_selection": True, "token_id": token_id}
            emotional_influence = self_model.get_emotional_influence_on_decision(emotional_decision_context)
            
            # Get current mood information
            current_mood = emotion_model.get_current_mood()
            mood_label = current_mood.get("mood_label", "neutral")
            valence = current_mood.get("valence", 0.0)
            arousal = current_mood.get("arousal", 0.3)
            
            # Base emotional weight (neutral)
            base_weight = 1.0
            influences = {}
            
            # All possible strategies
            all_strategies = {
                "belief_clarification", "fact_consolidation", "cluster_reassessment",
                "exploration_goal", "deep_exploration", "volatility_reduction",
                "conflict_resolution", "novelty_investigation"
            }
            
            # Mood-based strategy influences
            mood_strategy_influences = {
                "curious": {
                    "exploration_goal": 1.3,
                    "deep_exploration": 1.4,
                    "novelty_investigation": 1.2,
                    "belief_clarification": 1.1
                },
                "frustrated": {
                    "conflict_resolution": 1.3,
                    "volatility_reduction": 1.2,
                    "exploration_goal": 0.7,  # Avoid risky exploration when frustrated
                    "deep_exploration": 0.6
                },
                "calm": {
                    "fact_consolidation": 1.3,
                    "belief_clarification": 1.2,
                    "cluster_reassessment": 1.1,
                    "volatility_reduction": 1.2
                },
                "excited": {
                    "exploration_goal": 1.4,
                    "novelty_investigation": 1.3,
                    "deep_exploration": 1.2,
                    "fact_consolidation": 0.8  # Less patience for consolidation when excited
                },
                "content": {
                    "fact_consolidation": 1.2,
                    "belief_clarification": 1.1,
                    "exploration_goal": 1.0,
                    "conflict_resolution": 0.9
                },
                "tense": {
                    "conflict_resolution": 1.4,
                    "volatility_reduction": 1.3,
                    "belief_clarification": 1.1,
                    "exploration_goal": 0.6
                },
                "sad": {
                    "fact_consolidation": 1.1,
                    "cluster_reassessment": 1.0,
                    "exploration_goal": 0.7,
                    "deep_exploration": 0.6
                }
            }
            
            # Apply emotional influences for each strategy
            for strategy in all_strategies:
                weight = base_weight
                
                # Apply mood-specific influences
                if mood_label in mood_strategy_influences:
                    mood_influences = mood_strategy_influences[mood_label]
                    if strategy in mood_influences:
                        mood_multiplier = mood_influences[strategy]
                        weight = mood_multiplier
                
                # Apply general emotional characteristics
                risk_tolerance = emotional_influence.get("risk_tolerance", 0.5)
                exploration_bias = emotional_influence.get("exploration_bias", 0.5)
                patience = emotional_influence.get("patience", 0.5)
                novelty_seeking = emotional_influence.get("novelty_seeking", 0.5)
                
                # Adjust strategies based on emotional characteristics
                if strategy in ["exploration_goal", "deep_exploration", "novelty_investigation"]:
                    # Exploration strategies benefit from high exploration bias and novelty seeking
                    exploration_factor = (exploration_bias + novelty_seeking) / 2.0
                    weight *= (0.7 + exploration_factor * 0.6)  # Range: 0.7 to 1.3
                
                elif strategy in ["fact_consolidation", "cluster_reassessment"]:
                    # Consolidation strategies benefit from high patience
                    patience_factor = patience
                    weight *= (0.8 + patience_factor * 0.4)  # Range: 0.8 to 1.2
                
                elif strategy in ["conflict_resolution", "volatility_reduction"]:
                    # Conflict resolution benefits from balanced emotional state
                    stability_factor = 1.0 - abs(valence) - (arousal - 0.3)  # Prefer moderate arousal
                    stability_factor = max(0.0, min(1.0, stability_factor))
                    weight *= (0.8 + stability_factor * 0.4)  # Range: 0.8 to 1.2
                
                elif strategy in ["belief_clarification"]:
                    # Belief clarification is generally good but benefits from patience and stability
                    clarity_factor = (patience + (1.0 - abs(valence))) / 2.0
                    weight *= (0.9 + clarity_factor * 0.3)  # Range: 0.9 to 1.2
                
                # Constrain weight to reasonable range
                influences[strategy] = max(0.5, min(1.5, weight))
            
            logger.debug(f"[DriftExecutionEngine] Emotional influences for mood '{mood_label}': "
                        f"risk_tolerance={risk_tolerance:.2f}, exploration_bias={exploration_bias:.2f}, "
                        f"patience={patience:.2f}")
            
            return influences
            
        except Exception as e:
            logger.error(f"[DriftExecutionEngine] Error calculating emotional influences: {e}")
            # Return neutral influences if emotion system not available
            return {strategy: 1.0 for strategy in [
                "belief_clarification", "fact_consolidation", "cluster_reassessment",
                "exploration_goal", "deep_exploration", "volatility_reduction",
                "conflict_resolution", "novelty_investigation"
            ]}
    
    def check_drive_tensions(self) -> List[Tuple[int, float]]:
        """
        Check for tokens with high drive tension that may need autonomous attention.
        
        Returns list of (token_id, tension_score) for tokens above threshold.
        """
        try:
            # Get tokens ranked by drive pressure
            pressured_tokens = self.drive_system.rank_tokens_by_drive_pressure(20)
            
            # Filter by tension threshold
            high_tension_tokens = [
                (token_id, pressure) for token_id, pressure in pressured_tokens
                if pressure >= self.drive_tension_threshold
            ]
            
            return high_tension_tokens
            
        except Exception as e:
            logger.error(f"Failed to check drive tensions: {e}")
            return []
    
    def spawn_drive_goals_if_needed(self) -> List[Any]:
        """
        Check drive tensions and spawn autonomous goals if thresholds exceeded.
        
        Returns list of spawned goals.
        """
        try:
            spawned_goals = []
            high_tension_tokens = self.check_drive_tensions()
            
            for token_id, tension in high_tension_tokens:
                # Check if we already have recent goals for this token
                recent_goals = [
                    goal for goal in self.evolved_goals 
                    if (hasattr(goal, 'token_id') and goal.token_id == token_id and 
                        (time.time() - goal.created_at) < 3600)  # Within last hour
                ]
                
                if len(recent_goals) < 2:  # Limit concurrent goals per token
                    # Try to spawn a motivational goal
                    goal = self.drive_system.spawn_goal_if_needed(token_id)
                    if goal:
                        spawned_goals.append(goal)
                        logger.info(f"[DriftExecutionEngine] Spawned drive goal for token {token_id} "
                                  f"(tension: {tension:.3f})")
            
            return spawned_goals
            
        except Exception as e:
            logger.error(f"Failed to spawn drive goals: {e}")
            return []


# Global drift execution engine instance
_execution_engine_instance = None


def get_drift_execution_engine(memory_system=None) -> DriftExecutionEngine:
    """Get or create the global drift execution engine instance."""
    global _execution_engine_instance
    
    if _execution_engine_instance is None:
        _execution_engine_instance = DriftExecutionEngine(memory_system)
    
    return _execution_engine_instance


def start_drift_execution(memory_system=None):
    """Start automatic drift goal execution."""
    engine = get_drift_execution_engine(memory_system)
    engine.start_background_execution()


def stop_drift_execution():
    """Stop automatic drift goal execution."""
    global _execution_engine_instance
    
    if _execution_engine_instance:
        _execution_engine_instance.stop_background_execution()


def get_execution_status() -> Dict[str, Any]:
    """Get drift execution status."""
    engine = get_drift_execution_engine()
    return engine.get_execution_status() 