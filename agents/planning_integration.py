#!/usr/bin/env python3
"""
Planning Integration Module - Phase 11 & 13 Integration Hooks

Provides integration utilities for connecting the recursive planning system
with MeRNSTA's existing processing pipeline.

Phase 13 Enhancement: Added autonomous command generation and execution from planning insights.
"""

import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime

# Phase 13: Import command routing capabilities
try:
    from .command_router import route_command_async
    from config.settings import get_config
    COMMAND_ROUTER_AVAILABLE = True
except ImportError:
    COMMAND_ROUTER_AVAILABLE = False
    logging.warning("[PlanningIntegrator] Command router not available - autonomous commands disabled")

# Import the recursive planning components
try:
    from .recursive_planner import RecursivePlanner
    from .self_prompter import SelfPromptGenerator
    from storage.plan_memory import PlanMemory
    from storage.intention_model import IntentionModel, record_goal_intention
    RECURSIVE_PLANNING_AVAILABLE = True
except ImportError:
    RECURSIVE_PLANNING_AVAILABLE = False
    logging.warning("Recursive planning components not available")

class PlanningIntegrator:
    """
    Integration utility for connecting recursive planning with existing systems.
    
    Provides:
    - Goal detection in user input
    - Automatic plan generation for detected goals
    - Memory integration with intention tracking
    - Autonomous goal generation triggers
    """
    
    def __init__(self):
        self.enabled = RECURSIVE_PLANNING_AVAILABLE
        self._planner = None
        self._self_prompter = None
        self._plan_memory = None
        self._intention_model = None
        
        # Goal detection patterns
        self.goal_patterns = [
            r'\b(?:help me|teach me|show me how to|explain how to|guide me through)\b',
            r'\b(?:i want to|i need to|i would like to|my goal is to)\b',
            r'\b(?:create|build|make|develop|implement|design)\b.*\b(?:system|app|feature|tool)\b',
            r'\b(?:improve|optimize|enhance|upgrade|fix)\b',
            r'\b(?:learn|understand|master|study)\b.*\b(?:about|how to)\b',
            r'\b(?:solve|resolve|address|handle)\b.*\b(?:problem|issue|challenge)\b',
            r'\b(?:plan|strategy|approach|method)\b.*\b(?:for|to)\b'
        ]
        
        # Phase 13: Command generation configuration
        self.command_config = get_config().get('autonomous_commands', {}) if COMMAND_ROUTER_AVAILABLE else {}
        self.enable_planning_commands = self.command_config.get('enable_planning_commands', False)
        self.max_commands_per_plan = self.command_config.get('max_commands_per_cycle', 5)
        self.require_confirmation = self.command_config.get('require_confirmation', True)
        
        # Command generation patterns for planning
        self.planning_command_triggers = {
            'implementation': r'(?i)(implement|build|create|develop|code)',
            'testing': r'(?i)(test|verify|check|validate)',
            'documentation': r'(?i)(document|write|explain|describe)',
            'research': r'(?i)(research|investigate|explore|find out)',
            'installation': r'(?i)(install|setup|configure|deploy)',
            'optimization': r'(?i)(optimize|improve|enhance|speed up)',
        }
        
        logging.info(f"[PlanningIntegrator] Initialized (enabled: {self.enabled}, commands: {self.enable_planning_commands})")
    
    @property
    def planner(self):
        """Lazy-load recursive planner"""
        if not self.enabled:
            return None
        if self._planner is None:
            self._planner = RecursivePlanner()
        return self._planner
    
    @property
    def self_prompter(self):
        """Lazy-load self-prompter"""
        if not self.enabled:
            return None
        if self._self_prompter is None:
            self._self_prompter = SelfPromptGenerator()
        return self._self_prompter
    
    @property
    def plan_memory(self):
        """Lazy-load plan memory"""
        if not self.enabled:
            return None
        if self._plan_memory is None:
            self._plan_memory = PlanMemory()
        return self._plan_memory
    
    @property
    def intention_model(self):
        """Lazy-load intention model"""
        if not self.enabled:
            return None
        if self._intention_model is None:
            self._intention_model = IntentionModel()
        return self._intention_model
    
    def detect_goal_request(self, user_input: str) -> bool:
        """
        Detect if user input contains a goal or request for planning.
        
        Args:
            user_input: User's message
            
        Returns:
            True if input appears to be a goal request
        """
        if not self.enabled or not user_input:
            return False
        
        user_input_lower = user_input.lower()
        
        # Check against goal detection patterns
        for pattern in self.goal_patterns:
            if re.search(pattern, user_input_lower):
                return True
        
        # Additional heuristics
        # Length-based: longer inputs more likely to contain goals
        if len(user_input.split()) > 8:
            goal_keywords = ['want', 'need', 'goal', 'objective', 'plan', 'create', 'build', 'improve', 'learn']
            if any(keyword in user_input_lower for keyword in goal_keywords):
                return True
        
        # Question-based goals
        if user_input.strip().endswith('?') and any(word in user_input_lower for word in ['how', 'what', 'can you help']):
            return True
        
        return False
    
    def process_goal_request(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a detected goal request through the recursive planning system.
        
        Args:
            user_input: User's goal request
            context: Additional context for planning
            
        Returns:
            Processing results with plan information
        """
        if not self.enabled:
            return {"error": "Recursive planning not available"}
        
        try:
            # Extract goal text (clean up the input)
            goal_text = self._extract_goal_text(user_input)
            
            # Record intention for this goal
            goal_id = f"user-goal-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            if self.intention_model:
                intention_id = record_goal_intention(
                    goal_id=goal_id,
                    why=f"User requested: {goal_text}",
                    drive="user_request",
                    importance=0.8
                )
            
            # Generate plan
            plan = self.planner.plan_goal(goal_text, context=context)
            
            # Score the plan
            score = self.planner.score_plan(plan)
            
            result = {
                "goal_detected": True,
                "goal_text": goal_text,
                "goal_id": goal_id,
                "plan": plan,
                "plan_score": score,
                "auto_executable": score >= 0.8,  # High confidence plans can auto-execute
                "suggestion": self._generate_suggestion(plan, score)
            }
            
            # Add intention tracing if available
            if self.intention_model and intention_id:
                result["intention_chain"] = self.intention_model.trace_why_formatted(goal_id)
            
            logging.info(f"[PlanningIntegrator] Processed goal: {goal_text} (score: {score:.2f})")
            
            return result
            
        except Exception as e:
            logging.error(f"[PlanningIntegrator] Error processing goal request: {e}")
            return {"error": str(e), "goal_detected": True}
    
    def enhance_memory_with_intention(self, fact_data: Dict[str, Any], goal_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhance memory storage with intention tracking.
        
        Args:
            fact_data: Data being stored in memory
            goal_context: Context about why this fact is being stored
            
        Returns:
            Enhanced fact data with intention tracking
        """
        if not self.enabled or not goal_context:
            return fact_data
        
        try:
            # Record intention for memory storage
            if self.intention_model and goal_context.get('goal_id'):
                intention_id = self.intention_model.record_intention(
                    goal_id=goal_context['goal_id'],
                    triggered_by=goal_context.get('triggered_by', 'memory_operation'),
                    drive=goal_context.get('drive', 'knowledge_acquisition'),
                    importance=goal_context.get('importance', 0.5),
                    reflection_note=goal_context.get('reasoning', 'Storing knowledge for future reference')
                )
                
                # Add intention metadata to fact
                fact_data['intention_id'] = intention_id
                fact_data['goal_context'] = goal_context
            
            return fact_data
            
        except Exception as e:
            logging.warning(f"[PlanningIntegrator] Could not enhance memory with intention: {e}")
            return fact_data
    
    def trigger_autonomous_improvement(self, performance_data: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Trigger autonomous goal generation based on system performance.
        
        Args:
            performance_data: Optional performance metrics to guide generation
            
        Returns:
            List of generated self-improvement goals
        """
        if not self.enabled or not self.self_prompter:
            return []
        
        try:
            # Generate autonomous goals
            context = {"performance_data": performance_data} if performance_data else None
            goals = self.self_prompter.propose_goals(context)
            
            # Prioritize goals
            prioritized_goals = self.self_prompter.prioritize_goals(goals)
            
            logging.info(f"[PlanningIntegrator] Generated {len(prioritized_goals)} autonomous improvement goals")
            
            return prioritized_goals
            
        except Exception as e:
            logging.error(f"[PlanningIntegrator] Error in autonomous improvement: {e}")
            return []
    
    def check_goal_completion(self, goal_id: str, success: bool, outcome_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Mark a goal as completed and record the outcome.
        
        Args:
            goal_id: Goal identifier
            success: Whether goal was completed successfully
            outcome_data: Additional outcome information
            
        Returns:
            True if recorded successfully
        """
        if not self.enabled:
            return False
        
        try:
            # Update intention status
            if self.intention_model:
                if success:
                    self.intention_model.mark_goal_fulfilled(goal_id)
                else:
                    self.intention_model.mark_goal_abandoned(goal_id)
            
            # Record plan outcome if there's a related plan
            if self.plan_memory and outcome_data and 'plan_id' in outcome_data:
                self.plan_memory.record_plan_outcome(outcome_data['plan_id'], outcome_data)
            
            logging.info(f"[PlanningIntegrator] Recorded goal completion: {goal_id} (success: {success})")
            
            return True
            
        except Exception as e:
            logging.error(f"[PlanningIntegrator] Error recording goal completion: {e}")
            return False
    
    def _extract_goal_text(self, user_input: str) -> str:
        """Extract clean goal text from user input"""
        # Remove common prefixes
        goal_text = user_input.strip()
        
        prefixes_to_remove = [
            r'^(?:please\s+)?(?:help me|teach me|show me how to|explain how to|guide me through)\s+',
            r'^(?:i want to|i need to|i would like to|my goal is to)\s+',
            r'^(?:can you\s+)?(?:help me\s+)?',
        ]
        
        for prefix in prefixes_to_remove:
            goal_text = re.sub(prefix, '', goal_text, flags=re.IGNORECASE).strip()
        
        # Capitalize first letter
        if goal_text:
            goal_text = goal_text[0].upper() + goal_text[1:]
        
        return goal_text
    
    def _generate_suggestion(self, plan, score: float) -> str:
        """Generate a suggestion based on plan quality"""
        if score >= 0.8:
            return f"High confidence plan ready for execution. Use 'execute_plan {plan.plan_id}' to proceed."
        elif score >= 0.6:
            return f"Good plan generated. Review with 'show_plan {plan.plan_id}' before execution."
        elif score >= 0.4:
            return f"Plan needs refinement. Consider breaking down complex steps or gathering more information."
        else:
            return f"Plan quality is low. Consider providing more context or trying a different approach."
    
    # === PHASE 13: AUTONOMOUS COMMAND GENERATION FROM PLANS ===
    
    async def generate_plan_commands(self, plan_id: str) -> List[Dict[str, Any]]:
        """
        Generate actionable commands from a plan.
        
        Args:
            plan_id: ID of the plan to generate commands for
            
        Returns:
            List of command dictionaries to execute
        """
        if not self.enable_planning_commands or not COMMAND_ROUTER_AVAILABLE:
            return []
        
        try:
            if not self.plan_memory:
                return []
            
            # Get the plan
            plan = self.plan_memory.get_plan(plan_id)
            if not plan:
                logging.warning(f"[PlanningIntegrator] Plan {plan_id} not found")
                return []
            
            commands = []
            
            # Analyze plan steps for command opportunities
            for step in plan.steps:
                step_commands = await self._generate_step_commands(step, plan)
                commands.extend(step_commands)
            
            # Limit commands per plan
            commands = commands[:self.max_commands_per_plan]
            
            logging.info(f"[PlanningIntegrator] Generated {len(commands)} commands for plan {plan_id}")
            return commands
            
        except Exception as e:
            logging.error(f"[PlanningIntegrator] Error generating plan commands: {e}")
            return []
    
    async def _generate_step_commands(self, step: Dict[str, Any], plan: Any) -> List[Dict[str, Any]]:
        """Generate commands for a specific plan step."""
        commands = []
        step_description = step.get('description', '')
        step_action = step.get('action', '')
        
        # Combine description and action for analysis
        step_text = f"{step_description} {step_action}".lower()
        
        try:
            # Check for implementation commands
            if re.search(self.planning_command_triggers['implementation'], step_text):
                commands.extend(self._generate_implementation_commands(step, plan))
            
            # Check for testing commands
            if re.search(self.planning_command_triggers['testing'], step_text):
                commands.extend(self._generate_testing_commands(step, plan))
            
            # Check for documentation commands
            if re.search(self.planning_command_triggers['documentation'], step_text):
                commands.extend(self._generate_documentation_commands(step, plan))
            
            # Check for research commands
            if re.search(self.planning_command_triggers['research'], step_text):
                commands.extend(self._generate_research_commands(step, plan))
            
            # Check for installation commands
            if re.search(self.planning_command_triggers['installation'], step_text):
                commands.extend(self._generate_installation_commands(step, plan))
            
            # Check for optimization commands
            if re.search(self.planning_command_triggers['optimization'], step_text):
                commands.extend(self._generate_optimization_commands(step, plan))
            
        except Exception as e:
            logging.error(f"[PlanningIntegrator] Error generating step commands: {e}")
        
        return commands
    
    def _generate_implementation_commands(self, step: Dict[str, Any], plan: Any) -> List[Dict[str, Any]]:
        """Generate implementation commands for a plan step."""
        commands = []
        step_text = step.get('description', '').lower()
        
        # File creation commands
        if 'file' in step_text or 'create' in step_text:
            if 'python' in step_text:
                commands.append({
                    'command': '/run_tool write_file new_module.py "# New Python module\\npass"',
                    'purpose': f'Create Python file for step: {step.get("description", "")}',
                    'priority': 'high',
                    'step_id': step.get('step_id', ''),
                    'plan_id': plan.plan_id if hasattr(plan, 'plan_id') else ''
                })
        
        # Directory structure commands
        if 'directory' in step_text or 'folder' in step_text:
            commands.append({
                'command': '/run_shell "mkdir -p project_structure"',
                'purpose': f'Create directory structure for step: {step.get("description", "")}',
                'priority': 'medium',
                'step_id': step.get('step_id', ''),
                'plan_id': plan.plan_id if hasattr(plan, 'plan_id') else ''
            })
        
        return commands
    
    def _generate_testing_commands(self, step: Dict[str, Any], plan: Any) -> List[Dict[str, Any]]:
        """Generate testing commands for a plan step."""
        commands = []
        
        commands.append({
            'command': '/run_shell "python -m pytest tests/ -v --tb=short"',
            'purpose': f'Run tests for step: {step.get("description", "")}',
            'priority': 'high',
            'step_id': step.get('step_id', ''),
            'plan_id': plan.plan_id if hasattr(plan, 'plan_id') else ''
        })
        
        return commands
    
    def _generate_documentation_commands(self, step: Dict[str, Any], plan: Any) -> List[Dict[str, Any]]:
        """Generate documentation commands for a plan step."""
        commands = []
        
        commands.append({
            'command': '/run_tool read_file README.md',
            'purpose': f'Read existing documentation for step: {step.get("description", "")}',
            'priority': 'medium',
            'step_id': step.get('step_id', ''),
            'plan_id': plan.plan_id if hasattr(plan, 'plan_id') else ''
        })
        
        return commands
    
    def _generate_research_commands(self, step: Dict[str, Any], plan: Any) -> List[Dict[str, Any]]:
        """Generate research commands for a plan step."""
        commands = []
        step_text = step.get('description', '').lower()
        
        if 'documentation' in step_text:
            commands.append({
                'command': '/run_tool list_directory docs/',
                'purpose': f'Research documentation for step: {step.get("description", "")}',
                'priority': 'medium',
                'step_id': step.get('step_id', ''),
                'plan_id': plan.plan_id if hasattr(plan, 'plan_id') else ''
            })
        
        return commands
    
    def _generate_installation_commands(self, step: Dict[str, Any], plan: Any) -> List[Dict[str, Any]]:
        """Generate installation commands for a plan step."""
        commands = []
        step_text = step.get('description', '').lower()
        
        # Package installation
        if 'package' in step_text or 'library' in step_text:
            if 'requests' in step_text:
                commands.append({
                    'command': '/pip_install requests',
                    'purpose': f'Install requests package for step: {step.get("description", "")}',
                    'priority': 'high',
                    'step_id': step.get('step_id', ''),
                    'plan_id': plan.plan_id if hasattr(plan, 'plan_id') else ''
                })
            elif 'numpy' in step_text:
                commands.append({
                    'command': '/pip_install numpy',
                    'purpose': f'Install numpy package for step: {step.get("description", "")}',
                    'priority': 'high',
                    'step_id': step.get('step_id', ''),
                    'plan_id': plan.plan_id if hasattr(plan, 'plan_id') else ''
                })
        
        return commands
    
    def _generate_optimization_commands(self, step: Dict[str, Any], plan: Any) -> List[Dict[str, Any]]:
        """Generate optimization commands for a plan step."""
        commands = []
        
        commands.append({
            'command': '/run_shell "python -m py_compile *.py"',
            'purpose': f'Check Python syntax for optimization step: {step.get("description", "")}',
            'priority': 'medium',
            'step_id': step.get('step_id', ''),
            'plan_id': plan.plan_id if hasattr(plan, 'plan_id') else ''
        })
        
        return commands
    
    async def execute_plan_commands(self, plan_id: str) -> Dict[str, Any]:
        """
        Execute all commands generated for a plan.
        
        Args:
            plan_id: ID of the plan to execute commands for
            
        Returns:
            Execution results and summary
        """
        if not COMMAND_ROUTER_AVAILABLE:
            return {'error': 'Command router not available'}
        
        # Generate commands for the plan
        commands = await self.generate_plan_commands(plan_id)
        
        if not commands:
            return {'message': 'No commands generated for this plan'}
        
        execution_results = {
            'plan_id': plan_id,
            'commands_executed': 0,
            'commands_successful': 0,
            'commands_failed': 0,
            'results': [],
            'summary': ''
        }
        
        try:
            for cmd_info in commands:
                command = cmd_info['command']
                purpose = cmd_info.get('purpose', 'No purpose specified')
                priority = cmd_info.get('priority', 'medium')
                step_id = cmd_info.get('step_id', '')
                
                logging.info(f"[PlanningIntegrator] Executing plan command: {command} (step: {step_id})")
                
                # Execute command
                result = await route_command_async(command, "planning_integrator")
                
                execution_results['commands_executed'] += 1
                
                if result.get('success', False):
                    execution_results['commands_successful'] += 1
                    logging.info(f"[PlanningIntegrator] Command succeeded: {command}")
                else:
                    execution_results['commands_failed'] += 1
                    logging.warning(f"[PlanningIntegrator] Command failed: {command} - {result.get('error', 'Unknown error')}")
                
                execution_results['results'].append({
                    'command': command,
                    'purpose': purpose,
                    'priority': priority,
                    'step_id': step_id,
                    'success': result.get('success', False),
                    'output': result.get('output', ''),
                    'error': result.get('error', '')
                })
            
            # Generate summary
            success_rate = execution_results['commands_successful'] / max(execution_results['commands_executed'], 1)
            execution_results['summary'] = (
                f"Executed {execution_results['commands_executed']} plan commands. "
                f"Success rate: {success_rate:.1%} "
                f"({execution_results['commands_successful']} successful, {execution_results['commands_failed']} failed)"
            )
            
            logging.info(f"[PlanningIntegrator] {execution_results['summary']}")
            return execution_results
            
        except Exception as e:
            logging.error(f"[PlanningIntegrator] Error executing plan commands: {e}")
            execution_results['summary'] = f"Command execution error: {str(e)}"
            return execution_results


# Global integrator instance
_planning_integrator = None

def get_planning_integrator() -> PlanningIntegrator:
    """Get the global planning integrator instance"""
    global _planning_integrator
    if _planning_integrator is None:
        _planning_integrator = PlanningIntegrator()
    return _planning_integrator

# Convenience functions for easy integration

def is_goal_request(user_input: str) -> bool:
    """Check if user input is a goal request"""
    return get_planning_integrator().detect_goal_request(user_input)

def process_goal(user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process a goal through the planning system"""
    return get_planning_integrator().process_goal_request(user_input, context)

def enhance_memory_storage(fact_data: Dict[str, Any], goal_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Enhance memory storage with intention tracking"""
    return get_planning_integrator().enhance_memory_with_intention(fact_data, goal_context)

def generate_autonomous_goals(performance_data: Optional[Dict[str, Any]] = None) -> List[str]:
    """Generate autonomous improvement goals"""
    return get_planning_integrator().trigger_autonomous_improvement(performance_data)

def mark_goal_complete(goal_id: str, success: bool, outcome_data: Optional[Dict[str, Any]] = None) -> bool:
    """Mark a goal as completed"""
    return get_planning_integrator().check_goal_completion(goal_id, success, outcome_data)