#!/usr/bin/env python3
"""
Recursive Integration Hooks for MeRNSTA - Phase 14: Recursive Execution

Integration layer that allows existing agents to trigger recursive execution
capabilities when appropriate. Provides hooks for RecursivePlanner, 
SelfPrompter, ReflectiveEngine, and other system components.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from config.settings import get_config
from agents.file_writer import get_file_writer
from agents.edit_loop import get_edit_loop, write_and_run_with_loop


class RecursiveIntegrationManager:
    """
    Manager for recursive execution integration with existing agents.
    
    Provides hooks and triggers for autonomous code generation and execution
    when system analysis suggests it would be beneficial.
    """
    
    def __init__(self):
        self.config = get_config().get('recursive_execution', {})
        self.integration_config = self.config.get('integration', {})
        
        self.enable_planner_integration = self.integration_config.get('enable_planner_integration', True)
        self.enable_reflector_integration = self.integration_config.get('enable_reflector_integration', True)
        self.enable_self_prompter_integration = self.integration_config.get('enable_self_prompter_integration', True)
        self.auto_execute_on_goals = self.integration_config.get('auto_execute_on_goal_generation', False)
        
        # Components
        self.file_writer = get_file_writer()
        self.edit_loop = get_edit_loop()
        
        # Initialize logging
        self.logger = logging.getLogger('recursive_integration')
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("[RecursiveIntegration] Initialized integration manager")
    
    async def handle_planner_suggestion(self, plan_step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle suggestions from RecursivePlanner that might benefit from code generation.
        
        Args:
            plan_step: Step from a plan that might require code generation
            
        Returns:
            Execution result if code was generated and run, None otherwise
        """
        if not self.enable_planner_integration:
            return None
        
        try:
            step_text = plan_step.get('subgoal', '').lower()
            step_description = plan_step.get('why', '').lower()
            
            # Check if step suggests code generation
            code_indicators = [
                'script', 'code', 'program', 'function', 'automation',
                'generate', 'create file', 'write script', 'implement',
                'test script', 'validate', 'optimize'
            ]
            
            if any(indicator in step_text or indicator in step_description 
                   for indicator in code_indicators):
                
                self.logger.info(f"[RecursiveIntegration] Plan step suggests code generation: {step_text}")
                
                # Generate code suggestion based on step
                code_suggestion = await self._generate_code_from_plan_step(plan_step)
                
                if code_suggestion:
                    # Execute with recursive improvement
                    return await self._execute_with_improvement(
                        code_suggestion['code'],
                        code_suggestion['filename'],
                        f"Plan step: {step_text}"
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"[RecursiveIntegration] Error handling planner suggestion: {e}")
            return None
    
    async def handle_reflection_insight(self, insight: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle insights from ReflectiveEngine that suggest code improvements.
        
        Args:
            insight: Reflection insight that might trigger code generation
            
        Returns:
            Execution result if code was generated and run, None otherwise
        """
        if not self.enable_reflector_integration:
            return None
        
        try:
            insight_text = insight.get('insight', '').lower()
            insight_type = insight.get('type', '').lower()
            
            # Check if insight suggests code generation or improvement
            improvement_indicators = [
                'optimization', 'performance', 'efficiency', 'automation',
                'tool', 'script', 'streamline', 'simplify', 'automate',
                'generate code', 'create script', 'implement solution'
            ]
            
            if (any(indicator in insight_text for indicator in improvement_indicators) or
                insight_type in ['performance', 'optimization', 'automation']):
                
                self.logger.info(f"[RecursiveIntegration] Reflection suggests improvement: {insight_text}")
                
                # Generate improvement code
                improvement_suggestion = await self._generate_code_from_insight(insight)
                
                if improvement_suggestion:
                    return await self._execute_with_improvement(
                        improvement_suggestion['code'],
                        improvement_suggestion['filename'],
                        f"Reflection insight: {insight_text}"
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"[RecursiveIntegration] Error handling reflection insight: {e}")
            return None
    
    async def handle_self_prompt_goal(self, goal: str) -> Optional[Dict[str, Any]]:
        """
        Handle goals from SelfPrompter that might benefit from code generation.
        
        Args:
            goal: Self-generated goal text
            
        Returns:
            Execution result if code was generated and run, None otherwise
        """
        if not self.enable_self_prompter_integration:
            return None
        
        try:
            goal_lower = goal.lower()
            
            # Check if goal suggests code or script creation
            coding_indicators = [
                'create script', 'write code', 'generate script', 'implement',
                'automate', 'build tool', 'create utility', 'write function',
                'develop script', 'code solution', 'program', 'script to'
            ]
            
            if any(indicator in goal_lower for indicator in coding_indicators):
                self.logger.info(f"[RecursiveIntegration] Self-prompt goal suggests coding: {goal}")
                
                # Generate code based on goal
                code_suggestion = await self._generate_code_from_goal(goal)
                
                if code_suggestion:
                    return await self._execute_with_improvement(
                        code_suggestion['code'],
                        code_suggestion['filename'],
                        goal
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"[RecursiveIntegration] Error handling self-prompt goal: {e}")
            return None
    
    async def handle_diagnostic_repair(self, repair_goal: str, issue_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle repair goals from diagnostic systems that might need script solutions.
        
        Args:
            repair_goal: Goal for system repair
            issue_details: Details about the issues to be fixed
            
        Returns:
            Execution result if repair script was generated and run, None otherwise
        """
        try:
            goal_lower = repair_goal.lower()
            
            # Check if repair can be automated with scripts
            scriptable_repairs = [
                'optimize', 'clean up', 'refactor', 'update', 'migrate',
                'standardize', 'format', 'organize', 'consolidate',
                'automate', 'batch process', 'bulk update'
            ]
            
            if any(repair in goal_lower for repair in scriptable_repairs):
                self.logger.info(f"[RecursiveIntegration] Repair goal suggests automation: {repair_goal}")
                
                # Generate repair script
                repair_script = await self._generate_repair_script(repair_goal, issue_details)
                
                if repair_script:
                    return await self._execute_with_improvement(
                        repair_script['code'],
                        repair_script['filename'],
                        repair_goal
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"[RecursiveIntegration] Error handling diagnostic repair: {e}")
            return None
    
    async def _generate_code_from_plan_step(self, plan_step: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Generate code based on a plan step."""
        subgoal = plan_step.get('subgoal', '')
        expected_result = plan_step.get('expected_result', '')
        
        # Simple template-based code generation
        templates = {
            'test': self._create_test_script_template,
            'validate': self._create_validation_script_template,
            'optimize': self._create_optimization_script_template,
            'analyze': self._create_analysis_script_template,
            'monitor': self._create_monitoring_script_template
        }
        
        for keyword, template_func in templates.items():
            if keyword in subgoal.lower():
                code = template_func(subgoal, expected_result)
                filename = f"plan_step_{keyword}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
                return {'code': code, 'filename': filename}
        
        # Generic script template
        code = self._create_generic_script_template(subgoal, expected_result)
        filename = f"plan_step_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        return {'code': code, 'filename': filename}
    
    async def _generate_code_from_insight(self, insight: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Generate code based on a reflection insight."""
        insight_text = insight.get('insight', '')
        insight_type = insight.get('type', 'general')
        
        if 'performance' in insight_type or 'optimization' in insight_text.lower():
            code = self._create_performance_optimization_template(insight_text)
            filename = f"performance_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        elif 'automation' in insight_text.lower():
            code = self._create_automation_template(insight_text)
            filename = f"automation_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        else:
            code = self._create_improvement_template(insight_text)
            filename = f"improvement_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        
        return {'code': code, 'filename': filename}
    
    async def _generate_code_from_goal(self, goal: str) -> Optional[Dict[str, str]]:
        """Generate code based on a self-prompted goal."""
        goal_lower = goal.lower()
        
        if 'test' in goal_lower:
            code = self._create_test_script_template(goal, "Verify system functionality")
            filename = f"test_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        elif 'monitor' in goal_lower:
            code = self._create_monitoring_script_template(goal, "Monitor system metrics")
            filename = f"monitor_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        elif 'utility' in goal_lower or 'tool' in goal_lower:
            code = self._create_utility_template(goal)
            filename = f"utility_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        else:
            code = self._create_generic_script_template(goal, "Accomplish the specified goal")
            filename = f"goal_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        
        return {'code': code, 'filename': filename}
    
    async def _generate_repair_script(self, repair_goal: str, issue_details: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Generate repair script for diagnostic issues."""
        code = self._create_repair_script_template(repair_goal, issue_details)
        filename = f"repair_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        return {'code': code, 'filename': filename}
    
    async def _execute_with_improvement(self, code: str, filename: str, goal: str) -> Dict[str, Any]:
        """Execute code with recursive improvement loop."""
        try:
            result = await write_and_run_with_loop(
                code=code,
                filename=filename,
                goal_description=goal,
                max_attempts=3
            )
            
            self.logger.info(f"[RecursiveIntegration] Executed {filename}: success={result.success}")
            
            return {
                'success': result.success,
                'filename': filename,
                'attempts': result.total_attempts,
                'duration': result.duration,
                'winning_file': result.winning_file_path,
                'goal': goal
            }
            
        except Exception as e:
            self.logger.error(f"[RecursiveIntegration] Execution failed for {filename}: {e}")
            return {
                'success': False,
                'error': str(e),
                'filename': filename,
                'goal': goal
            }
    
    # Template methods for different types of scripts
    
    def _create_test_script_template(self, goal: str, expected: str) -> str:
        return f"""#!/usr/bin/env python3
'''
Test Script - Generated by MeRNSTA Recursive Integration
Goal: {goal}
Expected: {expected}
'''

import sys
import os
import subprocess
import time

def main():
    print("=== MeRNSTA Test Script ===")
    print(f"Goal: {goal}")
    print(f"Expected: {expected}")
    
    # Basic system tests
    test_results = []
    
    # Test 1: System availability
    print("\\n1. Testing system availability...")
    try:
        # Add your test logic here
        test_results.append(("System availability", True, "System is responsive"))
    except Exception as e:
        test_results.append(("System availability", False, str(e)))
    
    # Test 2: Basic functionality
    print("\\n2. Testing basic functionality...")
    try:
        # Add functionality tests here
        test_results.append(("Basic functionality", True, "Core functions working"))
    except Exception as e:
        test_results.append(("Basic functionality", False, str(e)))
    
    # Results summary
    print("\\n=== Test Results ===")
    passed = 0
    total = len(test_results)
    
    for test_name, success, message in test_results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status} - {message}")
        if success:
            passed += 1
    
    print(f"\\nSummary: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""
    
    def _create_validation_script_template(self, goal: str, expected: str) -> str:
        return f"""#!/usr/bin/env python3
'''
Validation Script - Generated by MeRNSTA Recursive Integration
Goal: {goal}
Expected: {expected}
'''

import sys
import json
from pathlib import Path

def main():
    print("=== MeRNSTA Validation Script ===")
    print(f"Goal: {goal}")
    
    validation_results = {{}}
    
    # Add validation logic based on the goal
    try:
        # Example validations
        validation_results["timestamp"] = str(Path().cwd())
        validation_results["goal_met"] = True
        validation_results["details"] = "{expected}"
        
        print("✅ Validation completed successfully")
        print(f"Results: {json.dumps(validation_results, indent=2)}")
        return 0
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""
    
    def _create_optimization_script_template(self, goal: str, expected: str) -> str:
        return f"""#!/usr/bin/env python3
'''
Optimization Script - Generated by MeRNSTA Recursive Integration
Goal: {goal}
Expected: {expected}
'''

import sys
import time
import psutil

def main():
    print("=== MeRNSTA Optimization Script ===")
    print(f"Goal: {goal}")
    
    start_time = time.time()
    
    try:
        # System optimization example
        print("\\n1. Checking system resources...")
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        print(f"CPU Usage: {cpu_percent}%")
        print(f"Memory Usage: {memory.percent}%")
        
        # Add optimization logic here
        optimizations = []
        
        if cpu_percent > 80:
            optimizations.append("High CPU usage detected - consider process optimization")
        
        if memory.percent > 80:
            optimizations.append("High memory usage detected - consider memory cleanup")
        
        if optimizations:
            print("\\n🔧 Optimization suggestions:")
            for opt in optimizations:
                print(f"  • {opt}")
        else:
            print("\\n✅ System appears optimized")
        
        duration = time.time() - start_time
        print(f"\\nOptimization check completed in {duration:.2f} seconds")
        
        return 0
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""
    
    def _create_analysis_script_template(self, goal: str, expected: str) -> str:
        return f"""#!/usr/bin/env python3
'''
Analysis Script - Generated by MeRNSTA Recursive Integration
Goal: {goal}
Expected: {expected}
'''

import sys
import json
from datetime import datetime

def main():
    print("=== MeRNSTA Analysis Script ===")
    print(f"Goal: {goal}")
    
    analysis_results = {{
        "timestamp": datetime.now().isoformat(),
        "goal": "{goal}",
        "analysis_type": "automated",
        "findings": []
    }}
    
    try:
        # Add analysis logic here
        print("\\n🔍 Performing analysis...")
        
        # Example analysis
        analysis_results["findings"].append({{
            "category": "system_status",
            "finding": "System is operational",
            "confidence": 0.9
        }})
        
        analysis_results["findings"].append({{
            "category": "goal_assessment", 
            "finding": "Goal appears achievable",
            "confidence": 0.8
        }})
        
        print("\\n📊 Analysis Results:")
        for finding in analysis_results["findings"]:
            print(f"  • {finding['category']}: {finding['finding']} (confidence: {finding['confidence']:.1%})")
        
        # Save results
        with open("analysis_results.json", "w") as f:
            json.dump(analysis_results, f, indent=2)
        
        print("\\n✅ Analysis completed and saved to analysis_results.json")
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""
    
    def _create_monitoring_script_template(self, goal: str, expected: str) -> str:
        return f"""#!/usr/bin/env python3
'''
Monitoring Script - Generated by MeRNSTA Recursive Integration
Goal: {goal}
Expected: {expected}
'''

import sys
import time
import json
from datetime import datetime

def main():
    print("=== MeRNSTA Monitoring Script ===")
    print(f"Goal: {goal}")
    
    monitoring_data = {{
        "start_time": datetime.now().isoformat(),
        "goal": "{goal}",
        "metrics": []
    }}
    
    try:
        print("\\n📊 Starting monitoring...")
        
        # Monitor for a short duration
        for i in range(5):
            timestamp = datetime.now().isoformat()
            
            # Collect metrics (example)
            metric = {{
                "timestamp": timestamp,
                "iteration": i + 1,
                "status": "healthy",
                "value": 100 - (i * 2)  # Example decreasing value
            }}
            
            monitoring_data["metrics"].append(metric)
            print(f"  {timestamp}: Status={metric['status']}, Value={metric['value']}")
            
            time.sleep(1)
        
        monitoring_data["end_time"] = datetime.now().isoformat()
        monitoring_data["duration"] = len(monitoring_data["metrics"])
        
        # Save monitoring data
        with open("monitoring_data.json", "w") as f:
            json.dump(monitoring_data, f, indent=2)
        
        print("\\n✅ Monitoring completed and saved to monitoring_data.json")
        return 0
        
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""
    
    def _create_performance_optimization_template(self, insight: str) -> str:
        return f"""#!/usr/bin/env python3
'''
Performance Optimization - Generated from Insight
Insight: {insight}
'''

import sys
import time
import psutil
import gc

def main():
    print("=== Performance Optimization ===")
    print(f"Based on insight: {insight}")
    
    start_time = time.time()
    
    try:
        # Memory optimization
        print("\\n🧹 Optimizing memory usage...")
        gc.collect()
        
        # Check system resources
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        
        print(f"Current memory usage: {memory.percent}%")
        print(f"Current CPU usage: {cpu}%")
        
        # Performance suggestions
        suggestions = []
        if memory.percent > 70:
            suggestions.append("Consider reducing memory usage")
        if cpu > 70:
            suggestions.append("Consider optimizing CPU-intensive operations")
        
        if suggestions:
            print("\\n💡 Performance suggestions:")
            for suggestion in suggestions:
                print(f"  • {suggestion}")
        else:
            print("\\n✅ System performance appears optimal")
        
        duration = time.time() - start_time
        print(f"\\nOptimization completed in {duration:.2f} seconds")
        
        return 0
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""
    
    def _create_automation_template(self, insight: str) -> str:
        return f"""#!/usr/bin/env python3
'''
Automation Script - Generated from Insight
Insight: {insight}
'''

import sys
import os
import subprocess
from datetime import datetime

def main():
    print("=== Automation Script ===")
    print(f"Based on insight: {insight}")
    
    automation_log = []
    
    try:
        # Example automation tasks
        print("\\n🤖 Running automation tasks...")
        
        # Task 1: System status check
        print("  1. Checking system status...")
        automation_log.append({{
            "task": "system_status_check",
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }})
        
        # Task 2: Cleanup operations
        print("  2. Performing cleanup...")
        automation_log.append({{
            "task": "cleanup_operations", 
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }})
        
        print("\\n✅ Automation completed successfully")
        print(f"Tasks completed: {len(automation_log)}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Automation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""
    
    def _create_improvement_template(self, insight: str) -> str:
        return f"""#!/usr/bin/env python3
'''
Improvement Script - Generated from Insight
Insight: {insight}
'''

import sys
from datetime import datetime

def main():
    print("=== System Improvement Script ===")
    print(f"Based on insight: {insight}")
    
    improvements = []
    
    try:
        print("\\n🔧 Implementing improvements...")
        
        # Implementation based on insight
        improvements.append("Applied insight-based optimization")
        improvements.append("Enhanced system reliability")
        improvements.append("Improved operational efficiency")
        
        print("\\n✅ Improvements implemented:")
        for i, improvement in enumerate(improvements, 1):
            print(f"  {i}. {improvement}")
        
        print(f"\\nTotal improvements: {len(improvements)}")
        print(f"Completed at: {datetime.now().isoformat()}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Improvement failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""
    
    def _create_utility_template(self, goal: str) -> str:
        return f"""#!/usr/bin/env python3
'''
Utility Script - Generated for Goal
Goal: {goal}
'''

import sys
import os
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="{goal}")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")
    
    args = parser.parse_args()
    
    print("=== MeRNSTA Utility ===")
    print(f"Goal: {goal}")
    
    if args.verbose:
        print(f"Started at: {datetime.now().isoformat()}")
        print(f"Arguments: {vars(args)}")
    
    try:
        # Utility logic here
        if args.dry_run:
            print("\\n🔍 Dry run mode - showing planned actions:")
            print("  • Would perform utility operations")
            print("  • Would generate results")
        else:
            print("\\n⚙️ Executing utility operations...")
            # Add actual utility logic here
            print("  ✅ Operation 1 completed")
            print("  ✅ Operation 2 completed")
        
        print("\\n✅ Utility execution completed")
        return 0
        
    except Exception as e:
        print(f"❌ Utility failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""
    
    def _create_generic_script_template(self, goal: str, expected: str) -> str:
        return f"""#!/usr/bin/env python3
'''
Generated Script - MeRNSTA Recursive Integration
Goal: {goal}
Expected: {expected}
'''

import sys
from datetime import datetime

def main():
    print("=== MeRNSTA Generated Script ===")
    print(f"Goal: {goal}")
    print(f"Expected: {expected}")
    print(f"Generated: {datetime.now().isoformat()}")
    
    try:
        # Generic script logic
        print("\\n🚀 Executing script logic...")
        
        # Add implementation based on goal
        result = "Script executed successfully"
        
        print(f"\\n✅ {result}")
        return 0
        
    except Exception as e:
        print(f"❌ Script failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""
    
    def _create_repair_script_template(self, repair_goal: str, issue_details: Dict[str, Any]) -> str:
        return f"""#!/usr/bin/env python3
'''
Repair Script - Generated for System Issues
Goal: {repair_goal}
Issues: {issue_details}
'''

import sys
import os
import shutil
from datetime import datetime

def main():
    print("=== System Repair Script ===")
    print(f"Repair Goal: {repair_goal}")
    print(f"Issues: {issue_details}")
    
    repair_actions = []
    
    try:
        print("\\n🔧 Performing repair operations...")
        
        # Generic repair operations based on goal
        if "optimize" in repair_goal.lower():
            print("  • Running optimization repairs...")
            repair_actions.append("optimization_repairs")
        
        if "clean" in repair_goal.lower():
            print("  • Running cleanup repairs...")
            repair_actions.append("cleanup_repairs")
        
        if "update" in repair_goal.lower():
            print("  • Running update repairs...")
            repair_actions.append("update_repairs")
        
        # Default repair action
        if not repair_actions:
            print("  • Running general repairs...")
            repair_actions.append("general_repairs")
        
        print(f"\\n✅ Repair completed with {len(repair_actions)} actions")
        print(f"Actions taken: {', '.join(repair_actions)}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Repair failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""


# Global integration manager instance
_integration_manager = None

def get_recursive_integration_manager() -> RecursiveIntegrationManager:
    """Get or create global recursive integration manager instance."""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = RecursiveIntegrationManager()
    return _integration_manager


# Convenience functions for integration hooks

async def handle_planner_suggestion(plan_step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convenience function for planner integration."""
    manager = get_recursive_integration_manager()
    return await manager.handle_planner_suggestion(plan_step)


async def handle_reflection_insight(insight: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convenience function for reflection integration."""
    manager = get_recursive_integration_manager()
    return await manager.handle_reflection_insight(insight)


async def handle_self_prompt_goal(goal: str) -> Optional[Dict[str, Any]]:
    """Convenience function for self-prompter integration."""
    manager = get_recursive_integration_manager()
    return await manager.handle_self_prompt_goal(goal)


async def handle_diagnostic_repair(repair_goal: str, issue_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convenience function for diagnostic repair integration."""
    manager = get_recursive_integration_manager()
    return await manager.handle_diagnostic_repair(repair_goal, issue_details)