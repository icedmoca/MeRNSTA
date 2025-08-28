#!/usr/bin/env python3
"""
EditLoop Controller for MeRNSTA - Phase 14: Recursive Execution

Core recursive loop: Propose → Write → Run → Analyze → Rewrite → Retry
Orchestrates autonomous code generation, execution, and iterative improvement.
"""

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from config.settings import get_config
from agents.file_writer import get_file_writer
from agents.execution_monitor import get_execution_monitor, ExecutionMetrics
from storage.recursive_execution_log import get_recursive_execution_logger, ExecutionSession, ExecutionAttempt


@dataclass
class EditLoopAttempt:
    """Container for a single edit loop attempt."""
    attempt_number: int
    code_content: str
    filename: str
    write_result: Dict[str, Any]
    execution_result: Dict[str, Any]
    analysis_metrics: ExecutionMetrics
    timestamp: str
    improvements_made: List[str]
    next_suggestions: List[str]


@dataclass
class EditLoopResult:
    """Container for complete edit loop results."""
    success: bool
    final_attempt: Optional[EditLoopAttempt]
    all_attempts: List[EditLoopAttempt]
    total_attempts: int
    duration: float
    termination_reason: str
    winning_file_path: Optional[str]
    error: Optional[str]


class EditLoop:
    """
    Autonomous recursive code improvement loop.
    
    Features:
    - Iterative code generation and testing
    - Intelligent failure analysis and fixes
    - Configurable retry limits and termination conditions
    - Integration with reflection and planning engines
    - Comprehensive logging and memory storage
    """
    
    def __init__(self, llm_provider: Optional[Callable] = None):
        self.config = get_config().get('recursive_execution', {})
        self.max_attempts = self.config.get('max_attempts', 5)
        self.enable_memory_logging = self.config.get('enable_memory_logging', True)
        self.auto_trigger_reflection = self.config.get('auto_trigger_on_reflection', True)
        self.reroute_failures = self.config.get('reroute_failures_to_self_healer', True)
        
        # Components
        self.file_writer = get_file_writer()
        self.execution_monitor = get_execution_monitor()
        self.recursive_logger = get_recursive_execution_logger()
        self.llm_provider = llm_provider
        
        # Initialize logging
        self.logger = logging.getLogger('edit_loop')
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("[EditLoop] Initialized recursive improvement controller")
    
    async def run_loop(self, initial_code: str, filename: str,
                      goal_description: str = None,
                      expected_behavior: str = None,
                      custom_validator: Optional[Callable] = None,
                      max_attempts: Optional[int] = None) -> EditLoopResult:
        """
        Run the complete edit loop with recursive improvement.
        
        Args:
            initial_code: Starting code content
            filename: Target filename
            goal_description: Description of what the code should accomplish
            expected_behavior: Expected execution behavior
            custom_validator: Optional custom validation function
            max_attempts: Override default max attempts
            
        Returns:
            EditLoopResult with complete loop results
        """
        start_time = datetime.now()
        attempts = []
        current_code = initial_code
        max_attempts = max_attempts or self.max_attempts
        
        # Generate unique session ID
        session_id = f"edit_loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(initial_code) % 10000}"
        
        self.logger.info(f"[EditLoop] Starting recursive loop for {filename} (session: {session_id})")
        
        try:
            for attempt_num in range(1, max_attempts + 1):
                self.logger.info(f"[EditLoop] Attempt {attempt_num}/{max_attempts}")
                
                # Execute current attempt
                attempt = await self._execute_attempt(
                    attempt_num, current_code, filename, expected_behavior, custom_validator
                )
                attempts.append(attempt)
                
                # Log attempt to recursive logger
                if self.enable_memory_logging:
                    await self._log_attempt_to_recursive_logger(session_id, attempt)
                
                # Check if successful
                if attempt.analysis_metrics.overall_success:
                    if custom_validator:
                        try:
                            if await self._run_custom_validator(custom_validator, attempt):
                                result = self._create_success_result(attempts, attempt, start_time)
                                if self.enable_memory_logging:
                                    await self._log_session_to_recursive_logger(session_id, initial_code, filename, goal_description, start_time, result)
                                return result
                        except Exception as e:
                            self.logger.warning(f"[EditLoop] Custom validator failed: {e}")
                    else:
                        result = self._create_success_result(attempts, attempt, start_time)
                        if self.enable_memory_logging:
                            await self._log_session_to_recursive_logger(session_id, initial_code, filename, goal_description, start_time, result)
                        return result
                
                # Check if should continue
                if not self.execution_monitor.should_retry(
                    attempt.analysis_metrics, attempt_num, max_attempts
                ):
                    result = self._create_failure_result(
                        attempts, start_time, "No actionable improvements available"
                    )
                    if self.enable_memory_logging:
                        await self._log_session_to_recursive_logger(session_id, initial_code, filename, goal_description, start_time, result)
                    return result
                
                # Generate improved code for next attempt
                improved_code = await self._improve_code(
                    current_code, attempt, goal_description
                )
                
                if improved_code == current_code:
                    result = self._create_failure_result(
                        attempts, start_time, "Unable to generate improvements"
                    )
                    if self.enable_memory_logging:
                        await self._log_session_to_recursive_logger(session_id, initial_code, filename, goal_description, start_time, result)
                    return result
                
                current_code = improved_code
            
            # Max attempts reached
            result = self._create_failure_result(
                attempts, start_time, f"Maximum attempts ({max_attempts}) reached"
            )
            if self.enable_memory_logging:
                await self._log_session_to_recursive_logger(session_id, initial_code, filename, goal_description, start_time, result)
            return result
            
        except Exception as e:
            self.logger.error(f"[EditLoop] Loop execution error: {e}")
            result = self._create_failure_result(
                attempts, start_time, f"Loop execution error: {str(e)}"
            )
            if self.enable_memory_logging:
                await self._log_session_to_recursive_logger(session_id, initial_code, filename, goal_description, start_time, result)
            return result
    
    async def _execute_attempt(self, attempt_num: int, code: str, filename: str,
                              expected_behavior: str = None,
                              custom_validator: Optional[Callable] = None) -> EditLoopAttempt:
        """Execute a single attempt in the edit loop."""
        timestamp = datetime.now().isoformat()
        
        # Write file
        write_result = self.file_writer.write_file(
            content=code,
            filename=filename,
            executable=True,
            add_timestamp=(attempt_num > 1),  # Add timestamp for retries
            metadata={
                'attempt': attempt_num,
                'goal': 'recursive_improvement',
                'expected_behavior': expected_behavior
            }
        )
        
        if not write_result['success']:
            # Create minimal attempt for write failure
            return EditLoopAttempt(
                attempt_number=attempt_num,
                code_content=code,
                filename=filename,
                write_result=write_result,
                execution_result={'success': False, 'error': 'Write failed'},
                analysis_metrics=ExecutionMetrics(
                    exit_code=-1, duration=0.0, output_lines=0, error_lines=0,
                    success_indicators=[], failure_indicators=['Write failed'],
                    warnings=[], exceptions=[], overall_success=False,
                    confidence_score=0.0, improvement_suggestions=['Fix write error']
                ),
                timestamp=timestamp,
                improvements_made=[],
                next_suggestions=['Fix write error']
            )
        
        # Execute file
        execution_result = await self._execute_file(write_result['filepath'])
        
        # Analyze execution
        analysis_metrics = self.execution_monitor.analyze_execution(
            execution_result,
            file_path=write_result['filepath'],
            expected_behavior=expected_behavior
        )
        
        # Create attempt record
        attempt = EditLoopAttempt(
            attempt_number=attempt_num,
            code_content=code,
            filename=filename,
            write_result=write_result,
            execution_result=execution_result,
            analysis_metrics=analysis_metrics,
            timestamp=timestamp,
            improvements_made=[],  # Will be filled by improvement logic
            next_suggestions=analysis_metrics.improvement_suggestions
        )
        
        # Log attempt if enabled
        if self.enable_memory_logging:
            await self._log_attempt(attempt)
        
        return attempt
    
    async def _execute_file(self, file_path: str) -> Dict[str, Any]:
        """Execute a file and return results."""
        from agents.command_router import get_command_router
        
        # Determine execution command based on file extension
        if file_path.endswith('.py'):
            command = f'/run_shell "python3 {file_path}"'
        elif file_path.endswith('.sh'):
            command = f'/run_shell "bash {file_path}"'
        else:
            command = f'/run_shell "{file_path}"'
        
        # Execute through command router
        router = get_command_router()
        return await router.execute_command(command, "edit_loop")
    
    async def _improve_code(self, current_code: str, failed_attempt: EditLoopAttempt,
                           goal_description: str = None) -> str:
        """Generate improved code based on failure analysis."""
        try:
            # Collect improvement context
            context = {
                'current_code': current_code,
                'exit_code': failed_attempt.analysis_metrics.exit_code,
                'output': failed_attempt.execution_result.get('output', ''),
                'error': failed_attempt.execution_result.get('error', ''),
                'exceptions': failed_attempt.analysis_metrics.exceptions,
                'suggestions': failed_attempt.analysis_metrics.improvement_suggestions,
                'goal': goal_description
            }
            
            # Try different improvement strategies
            improved_code = None
            
            # Strategy 1: Use LLM provider if available
            if self.llm_provider:
                improved_code = await self._llm_improve_code(context)
            
            # Strategy 2: Pattern-based improvements
            if not improved_code or improved_code == current_code:
                improved_code = self._pattern_improve_code(context)
            
            # Strategy 3: Template-based fixes
            if not improved_code or improved_code == current_code:
                improved_code = self._template_improve_code(context)
            
            return improved_code or current_code
            
        except Exception as e:
            self.logger.error(f"[EditLoop] Error improving code: {e}")
            return current_code
    
    async def _llm_improve_code(self, context: Dict[str, Any]) -> Optional[str]:
        """Use LLM to improve code based on context."""
        if not self.llm_provider:
            return None
        
        try:
            prompt = self._build_improvement_prompt(context)
            response = await self.llm_provider(prompt)
            
            # Extract code from response (implementation depends on LLM provider)
            # This is a placeholder - actual implementation would parse LLM response
            return self._extract_code_from_response(response)
            
        except Exception as e:
            self.logger.error(f"[EditLoop] LLM improvement error: {e}")
            return None
    
    def _pattern_improve_code(self, context: Dict[str, Any]) -> Optional[str]:
        """Apply pattern-based improvements to code."""
        code = context['current_code']
        suggestions = context['suggestions']
        error = context['error']
        
        # Apply common fixes based on suggestions
        for suggestion in suggestions:
            suggestion_lower = suggestion.lower()
            
            # Fix missing imports
            if 'import' in suggestion_lower and 'missing' in suggestion_lower:
                if 'sys' in error.lower():
                    code = 'import sys\n' + code
                elif 'os' in error.lower():
                    code = 'import os\n' + code
                elif 'json' in error.lower():
                    code = 'import json\n' + code
            
            # Fix indentation
            if 'indentation' in suggestion_lower:
                lines = code.split('\n')
                fixed_lines = []
                for line in lines:
                    if line.strip():
                        # Simple indentation fix
                        if not line.startswith(' ') and not line.startswith('\t'):
                            if line.strip().startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'with ', 'try:')):
                                fixed_lines.append(line)
                            else:
                                fixed_lines.append('    ' + line.strip())
                        else:
                            fixed_lines.append(line)
                    else:
                        fixed_lines.append(line)
                code = '\n'.join(fixed_lines)
            
            # Fix syntax errors
            if 'syntax' in suggestion_lower:
                # Common syntax fixes
                code = code.replace('print ', 'print(').replace('\n)', ')\n')
                if not code.strip().endswith(')') and 'print(' in code:
                    code = code.rstrip() + ')'
        
        return code if code != context['current_code'] else None
    
    def _template_improve_code(self, context: Dict[str, Any]) -> Optional[str]:
        """Apply template-based improvements."""
        code = context['current_code']
        
        # Basic Python script template
        if context.get('goal') and 'python' in context['goal'].lower():
            if not code.startswith('#!'):
                template = '''#!/usr/bin/env python3

def main():
    """Main function."""
    # Your code here
    pass

if __name__ == "__main__":
    main()
'''
                # Try to embed existing code in template
                if 'def main' not in code:
                    indented_code = '\n'.join(f'    {line}' for line in code.split('\n') if line.strip())
                    return template.replace('    # Your code here\n    pass', indented_code)
        
        return None
    
    def _build_improvement_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for LLM-based code improvement."""
        return f"""
Improve the following code based on the execution results:

CURRENT CODE:
{context['current_code']}

EXECUTION RESULTS:
Exit Code: {context['exit_code']}
Output: {context.get('output', 'No output')}
Error: {context.get('error', 'No error')}

ISSUES FOUND:
{', '.join(context.get('exceptions', []))}

SUGGESTIONS:
{', '.join(context.get('suggestions', []))}

GOAL: {context.get('goal', 'Make the code work correctly')}

Please provide the improved code that fixes the identified issues.
"""
    
    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract code from LLM response."""
        # Simple extraction - look for code blocks
        import re
        
        # Look for code blocks
        code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # If no code blocks, return the response itself if it looks like code
        if any(keyword in response for keyword in ['def ', 'import ', 'print(', 'if __name__']):
            return response.strip()
        
        return None
    
    async def _run_custom_validator(self, validator: Callable, attempt: EditLoopAttempt) -> bool:
        """Run custom validation function."""
        if asyncio.iscoroutinefunction(validator):
            return await validator(attempt)
        else:
            return validator(attempt)
    
    async def _log_attempt(self, attempt: EditLoopAttempt):
        """Log attempt to memory and tool logger."""
        try:
            from storage.tool_use_log import get_tool_logger
            tool_logger = get_tool_logger()
            
            attempt_data = {
                'attempt_number': attempt.attempt_number,
                'filename': attempt.filename,
                'success': attempt.analysis_metrics.overall_success,
                'confidence': attempt.analysis_metrics.confidence_score,
                'exit_code': attempt.analysis_metrics.exit_code,
                'suggestions': attempt.next_suggestions
            }
            
            tool_logger.log_system_event(
                event_type='edit_loop_attempt',
                description=f"Edit loop attempt {attempt.attempt_number} for {attempt.filename}",
                data=attempt_data
            )
            
        except Exception as e:
            self.logger.error(f"[EditLoop] Error logging attempt: {e}")
    
    async def _log_attempt_to_recursive_logger(self, session_id: str, attempt: EditLoopAttempt):
        """Log attempt to recursive execution logger."""
        try:
            execution_attempt = ExecutionAttempt(
                session_id=session_id,
                attempt_number=attempt.attempt_number,
                code_content=attempt.code_content,
                write_success=attempt.write_result.get('success', False),
                execution_success=attempt.analysis_metrics.overall_success,
                exit_code=attempt.analysis_metrics.exit_code,
                output=attempt.execution_result.get('output', ''),
                error=attempt.execution_result.get('error', ''),
                duration=attempt.execution_result.get('duration', 0.0),
                confidence_score=attempt.analysis_metrics.confidence_score,
                improvement_suggestions=attempt.next_suggestions,
                timestamp=datetime.fromisoformat(attempt.timestamp)
            )
            
            self.recursive_logger.log_execution_attempt(execution_attempt)
            
            # Log performance metrics
            if attempt.execution_result.get('duration'):
                self.recursive_logger.log_performance_metric(
                    session_id, 'execution_duration', 
                    attempt.execution_result['duration'], 'seconds'
                )
            
            self.recursive_logger.log_performance_metric(
                session_id, 'confidence_score',
                attempt.analysis_metrics.confidence_score, 'ratio'
            )
            
        except Exception as e:
            self.logger.error(f"[EditLoop] Error logging attempt to recursive logger: {e}")
    
    async def _log_session_to_recursive_logger(self, session_id: str, initial_code: str, 
                                             filename: str, goal: str, start_time: datetime, 
                                             result: EditLoopResult):
        """Log complete session to recursive execution logger."""
        try:
            execution_session = ExecutionSession(
                session_id=session_id,
                goal=goal or "Recursive code improvement",
                initial_code=initial_code,
                filename=filename,
                start_time=start_time,
                end_time=datetime.now(),
                success=result.success,
                total_attempts=result.total_attempts,
                winning_attempt=result.final_attempt.attempt_number if result.final_attempt else None,
                termination_reason=result.termination_reason,
                duration=result.duration,
                metadata={
                    'winning_file_path': result.winning_file_path,
                    'error': result.error,
                    'attempt_count': len(result.all_attempts)
                }
            )
            
            self.recursive_logger.log_execution_session(execution_session)
            
            # Log code improvements between attempts
            for i in range(1, len(result.all_attempts)):
                prev_attempt = result.all_attempts[i-1]
                curr_attempt = result.all_attempts[i]
                
                if curr_attempt.code_content != prev_attempt.code_content:
                    improvement_type = self._classify_improvement(prev_attempt, curr_attempt)
                    effectiveness = curr_attempt.analysis_metrics.confidence_score - prev_attempt.analysis_metrics.confidence_score
                    
                    self.recursive_logger.log_code_improvement(
                        session_id,
                        prev_attempt.attempt_number,
                        curr_attempt.attempt_number,
                        improvement_type,
                        f"Code improved from attempt {prev_attempt.attempt_number} to {curr_attempt.attempt_number}",
                        effectiveness
                    )
            
        except Exception as e:
            self.logger.error(f"[EditLoop] Error logging session to recursive logger: {e}")
    
    def _classify_improvement(self, prev_attempt: EditLoopAttempt, curr_attempt: EditLoopAttempt) -> str:
        """Classify the type of improvement between attempts."""
        prev_suggestions = set(s.lower() for s in prev_attempt.next_suggestions)
        
        if any('syntax' in s for s in prev_suggestions):
            return 'syntax_fix'
        elif any('import' in s for s in prev_suggestions):
            return 'import_fix'
        elif any('indentation' in s for s in prev_suggestions):
            return 'formatting_fix'
        elif any('variable' in s or 'name' in s for s in prev_suggestions):
            return 'variable_fix'
        elif any('type' in s for s in prev_suggestions):
            return 'type_fix'
        else:
            return 'general_improvement'
    
    def _create_success_result(self, attempts: List[EditLoopAttempt], 
                             successful_attempt: EditLoopAttempt,
                             start_time: datetime) -> EditLoopResult:
        """Create success result."""
        duration = (datetime.now() - start_time).total_seconds()
        
        return EditLoopResult(
            success=True,
            final_attempt=successful_attempt,
            all_attempts=attempts,
            total_attempts=len(attempts),
            duration=duration,
            termination_reason="Success achieved",
            winning_file_path=successful_attempt.write_result.get('filepath'),
            error=None
        )
    
    def _create_failure_result(self, attempts: List[EditLoopAttempt],
                             start_time: datetime, reason: str) -> EditLoopResult:
        """Create failure result."""
        duration = (datetime.now() - start_time).total_seconds()
        
        return EditLoopResult(
            success=False,
            final_attempt=attempts[-1] if attempts else None,
            all_attempts=attempts,
            total_attempts=len(attempts),
            duration=duration,
            termination_reason=reason,
            winning_file_path=None,
            error=reason
        )


# Convenience functions

async def write_and_run_with_loop(code: str, filename: str, **kwargs) -> EditLoopResult:
    """
    Write code and run it with recursive improvement loop.
    
    Args:
        code: Initial code content
        filename: Target filename
        **kwargs: Additional arguments for run_loop
        
    Returns:
        EditLoopResult with complete loop results
    """
    loop = EditLoop()
    return await loop.run_loop(code, filename, **kwargs)


async def recursive_improve_file(file_path: str, goal_description: str = None, **kwargs) -> EditLoopResult:
    """
    Read existing file and improve it recursively.
    
    Args:
        file_path: Path to existing file
        goal_description: Description of improvement goal
        **kwargs: Additional arguments for run_loop
        
    Returns:
        EditLoopResult with improvement results
    """
    try:
        with open(file_path, 'r') as f:
            current_code = f.read()
        
        filename = Path(file_path).name
        loop = EditLoop()
        return await loop.run_loop(current_code, filename, goal_description, **kwargs)
        
    except Exception as e:
        return EditLoopResult(
            success=False,
            final_attempt=None,
            all_attempts=[],
            total_attempts=0,
            duration=0.0,
            termination_reason=f"File read error: {str(e)}",
            winning_file_path=None,
            error=str(e)
        )


# Global edit loop instance
_edit_loop = None

def get_edit_loop() -> EditLoop:
    """Get or create global edit loop instance."""
    global _edit_loop
    if _edit_loop is None:
        _edit_loop = EditLoop()
    return _edit_loop