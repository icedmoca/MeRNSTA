#!/usr/bin/env python3
"""
ExecutionMonitor Agent for MeRNSTA - Phase 14: Recursive Execution

Monitors and analyzes the execution of generated code files.
Provides intelligent success/failure detection and improvement suggestions.
"""

import re
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from config.settings import get_config


@dataclass
class ExecutionMetrics:
    """Container for execution analysis metrics."""
    exit_code: int
    duration: float
    output_lines: int
    error_lines: int
    success_indicators: List[str]
    failure_indicators: List[str]
    warnings: List[str]
    exceptions: List[str]
    overall_success: bool
    confidence_score: float
    improvement_suggestions: List[str]


class ExecutionMonitor:
    """
    Intelligent execution monitor with pattern-based analysis.
    
    Features:
    - Exit code analysis
    - Output pattern matching for success/failure
    - Exception detection and categorization
    - Performance metrics extraction
    - Improvement suggestion generation
    - Memory and logging integration
    """
    
    def __init__(self):
        self.config = get_config().get('recursive_execution', {})
        self.enable_memory_logging = self.config.get('enable_memory_logging', True)
        self.success_patterns = self._load_success_patterns()
        self.failure_patterns = self._load_failure_patterns()
        self.exception_patterns = self._load_exception_patterns()
        
        # Initialize logging
        self.logger = logging.getLogger('execution_monitor')
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("[ExecutionMonitor] Initialized with pattern-based analysis")
    
    def analyze_execution(self, execution_result: Dict[str, Any], 
                         file_path: str = None,
                         expected_behavior: str = None) -> ExecutionMetrics:
        """
        Analyze execution result and generate comprehensive metrics.
        
        Args:
            execution_result: Result from command router execution
            file_path: Path to executed file
            expected_behavior: Description of expected behavior
            
        Returns:
            ExecutionMetrics with detailed analysis
        """
        try:
            # Extract basic metrics
            exit_code = execution_result.get('exit_code', 0)
            duration = execution_result.get('duration', 0.0)
            output = execution_result.get('output', '')
            error = execution_result.get('error', '')
            
            # Analyze output and errors
            output_lines = len(output.split('\n')) if output else 0
            error_lines = len(error.split('\n')) if error else 0
            
            # Pattern matching
            success_indicators = self._find_success_patterns(output, error)
            failure_indicators = self._find_failure_patterns(output, error)
            warnings = self._find_warnings(output, error)
            exceptions = self._find_exceptions(output, error)
            
            # Calculate overall success and confidence
            overall_success, confidence_score = self._calculate_success_confidence(
                exit_code, success_indicators, failure_indicators, exceptions
            )
            
            # Generate improvement suggestions
            improvement_suggestions = self._generate_suggestions(
                exit_code, output, error, exceptions, expected_behavior
            )
            
            metrics = ExecutionMetrics(
                exit_code=exit_code,
                duration=duration,
                output_lines=output_lines,
                error_lines=error_lines,
                success_indicators=success_indicators,
                failure_indicators=failure_indicators,
                warnings=warnings,
                exceptions=exceptions,
                overall_success=overall_success,
                confidence_score=confidence_score,
                improvement_suggestions=improvement_suggestions
            )
            
            # Log to memory and tool logger if enabled
            if self.enable_memory_logging:
                self._log_execution_analysis(execution_result, metrics, file_path)
            
            self.logger.info(f"[ExecutionMonitor] Analyzed execution: success={overall_success}, confidence={confidence_score:.2f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"[ExecutionMonitor] Error analyzing execution: {e}")
            # Return minimal metrics on error
            return ExecutionMetrics(
                exit_code=execution_result.get('exit_code', -1),
                duration=execution_result.get('duration', 0.0),
                output_lines=0,
                error_lines=0,
                success_indicators=[],
                failure_indicators=[f"Analysis error: {str(e)}"],
                warnings=[],
                exceptions=[],
                overall_success=False,
                confidence_score=0.0,
                improvement_suggestions=["Fix analysis error and retry"]
            )
    
    def should_retry(self, metrics: ExecutionMetrics, attempt_count: int, 
                    max_attempts: int = 5) -> bool:
        """
        Determine if execution should be retried based on metrics.
        
        Args:
            metrics: Execution metrics from analysis
            attempt_count: Current attempt number
            max_attempts: Maximum allowed attempts
            
        Returns:
            True if should retry, False otherwise
        """
        # Don't retry if max attempts reached
        if attempt_count >= max_attempts:
            return False
        
        # Don't retry if clearly successful
        if metrics.overall_success and metrics.confidence_score > 0.8:
            return False
        
        # Retry if there are actionable improvement suggestions
        if metrics.improvement_suggestions:
            actionable_suggestions = [
                s for s in metrics.improvement_suggestions 
                if any(keyword in s.lower() for keyword in [
                    'import', 'syntax', 'indentation', 'missing', 'typo', 'variable'
                ])
            ]
            if actionable_suggestions:
                return True
        
        # Retry if failure seems fixable (not permanent errors)
        permanent_failures = [
            'permission denied',
            'file not found',
            'command not found',
            'no such file or directory'
        ]
        
        for indicator in metrics.failure_indicators:
            if any(pf in indicator.lower() for pf in permanent_failures):
                return False
        
        # Retry if exit code suggests temporary failure
        retriable_exit_codes = [1, 2, 126, 127]  # Common retriable exit codes
        if metrics.exit_code in retriable_exit_codes:
            return True
        
        return False
    
    def _load_success_patterns(self) -> List[str]:
        """Load success detection patterns."""
        return [
            r'success(fully)?',
            r'complete(d)?',
            r'finished',
            r'done',
            r'passed',
            r'ok(?:ay)?',
            r'✓|✅',
            r'test.*pass',
            r'all.*pass',
            r'no errors?',
            r'execution.*successful',
            r'process.*complete'
        ]
    
    def _load_failure_patterns(self) -> List[str]:
        """Load failure detection patterns."""
        return [
            r'error:?',
            r'failed?:?',
            r'exception:?',
            r'traceback',
            r'❌|✗',
            r'syntax error',
            r'import error',
            r'name error',
            r'type error',
            r'value error',
            r'file not found',
            r'permission denied',
            r'command not found',
            r'no such file',
            r'cannot access',
            r'test.*fail',
            r'assertion.*fail'
        ]
    
    def _load_exception_patterns(self) -> List[str]:
        """Load exception detection patterns."""
        return [
            r'Traceback \(most recent call last\):',
            r'\w+Error:',
            r'\w+Exception:',
            r'SyntaxError:',
            r'ImportError:',
            r'NameError:',
            r'TypeError:',
            r'ValueError:',
            r'AttributeError:',
            r'KeyError:',
            r'IndexError:',
            r'FileNotFoundError:',
            r'PermissionError:'
        ]
    
    def _find_success_patterns(self, output: str, error: str) -> List[str]:
        """Find success indicators in output."""
        indicators = []
        text = f"{output}\n{error}".lower()
        
        for pattern in self.success_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            indicators.extend(matches)
        
        return indicators
    
    def _find_failure_patterns(self, output: str, error: str) -> List[str]:
        """Find failure indicators in output."""
        indicators = []
        text = f"{output}\n{error}"
        
        for pattern in self.failure_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            indicators.extend(matches)
        
        return indicators
    
    def _find_warnings(self, output: str, error: str) -> List[str]:
        """Find warning messages in output."""
        warnings = []
        text = f"{output}\n{error}"
        
        warning_patterns = [
            r'warning:.*',
            r'deprecated:.*',
            r'caution:.*',
            r'⚠️.*'
        ]
        
        for pattern in warning_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            warnings.extend(matches)
        
        return warnings
    
    def _find_exceptions(self, output: str, error: str) -> List[str]:
        """Find exception details in output."""
        exceptions = []
        text = f"{output}\n{error}"
        
        for pattern in self.exception_patterns:
            matches = re.findall(pattern, text)
            exceptions.extend(matches)
        
        return exceptions
    
    def _calculate_success_confidence(self, exit_code: int, success_indicators: List[str],
                                    failure_indicators: List[str], exceptions: List[str]) -> Tuple[bool, float]:
        """Calculate overall success status and confidence score."""
        confidence = 0.5  # Start neutral
        
        # Exit code analysis
        if exit_code == 0:
            confidence += 0.3
        elif exit_code != 0:
            confidence -= 0.4
        
        # Success indicators boost confidence
        if success_indicators:
            confidence += min(0.3, len(success_indicators) * 0.1)
        
        # Failure indicators reduce confidence
        if failure_indicators:
            confidence -= min(0.5, len(failure_indicators) * 0.15)
        
        # Exceptions are strong failure indicators
        if exceptions:
            confidence -= min(0.4, len(exceptions) * 0.2)
        
        # Clamp confidence to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        # Determine overall success
        overall_success = confidence > 0.6 and exit_code == 0 and not exceptions
        
        return overall_success, confidence
    
    def _generate_suggestions(self, exit_code: int, output: str, error: str,
                            exceptions: List[str], expected_behavior: str = None) -> List[str]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []
        
        # Exit code specific suggestions
        if exit_code == 1:
            suggestions.append("General error - check output for specific issues")
        elif exit_code == 2:
            suggestions.append("Misuse of shell command - verify syntax")
        elif exit_code == 126:
            suggestions.append("Permission denied - check file permissions")
        elif exit_code == 127:
            suggestions.append("Command not found - verify executable path")
        elif exit_code == 130:
            suggestions.append("Script terminated by Ctrl+C")
        
        # Exception-based suggestions
        for exception in exceptions:
            if 'SyntaxError' in exception:
                suggestions.append("Fix syntax error - check indentation and punctuation")
            elif 'ImportError' in exception:
                suggestions.append("Fix import error - verify module names and availability")
            elif 'NameError' in exception:
                suggestions.append("Fix undefined variable - check variable names and scope")
            elif 'TypeError' in exception:
                suggestions.append("Fix type error - check function arguments and types")
            elif 'ValueError' in exception:
                suggestions.append("Fix value error - check input values and ranges")
            elif 'FileNotFoundError' in exception:
                suggestions.append("Fix file not found - verify file paths")
            elif 'PermissionError' in exception:
                suggestions.append("Fix permission error - check file/directory permissions")
        
        # Output-based suggestions
        error_lower = error.lower()
        
        if 'no module named' in error_lower:
            module_name = re.search(r"no module named ['\"]([^'\"]+)['\"]", error_lower)
            if module_name:
                suggestions.append(f"Install missing module: pip install {module_name.group(1)}")
        
        if 'command not found' in error_lower:
            suggestions.append("Install missing command or check PATH")
        
        if 'indentation' in error_lower:
            suggestions.append("Fix indentation - use consistent spaces or tabs")
        
        if 'unexpected indent' in error_lower:
            suggestions.append("Remove unexpected indentation")
        
        # Generic suggestions if no specific issues found
        if not suggestions and not exceptions and exit_code != 0:
            suggestions.append("Review output for clues about the issue")
            if expected_behavior:
                suggestions.append(f"Verify that output matches expected behavior: {expected_behavior}")
        
        return suggestions
    
    def _log_execution_analysis(self, execution_result: Dict[str, Any], 
                              metrics: ExecutionMetrics, file_path: str = None):
        """Log execution analysis to memory and tool logger."""
        try:
            # Log to tool use logger
            from storage.tool_use_log import get_tool_logger
            tool_logger = get_tool_logger()
            
            analysis_data = {
                'file_path': file_path,
                'exit_code': metrics.exit_code,
                'duration': metrics.duration,
                'overall_success': metrics.overall_success,
                'confidence_score': metrics.confidence_score,
                'success_indicators': metrics.success_indicators,
                'failure_indicators': metrics.failure_indicators,
                'exceptions': metrics.exceptions,
                'suggestions': metrics.improvement_suggestions
            }
            
            tool_logger.log_system_event(
                event_type='execution_analysis',
                description=f"Analyzed execution of {file_path or 'unknown file'}",
                data=analysis_data
            )
            
            # Log to memory system if available
            try:
                from memory.memory_store import get_memory_store
                memory = get_memory_store()
                
                memory.store_fact(
                    f"execution_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    json.dumps(analysis_data),
                    'execution_monitor',
                    {'type': 'execution_analysis', 'success': metrics.overall_success}
                )
            except ImportError:
                pass  # Memory system not available
                
        except Exception as e:
            self.logger.error(f"[ExecutionMonitor] Error logging analysis: {e}")


# Global execution monitor instance
_execution_monitor = None

def get_execution_monitor() -> ExecutionMonitor:
    """Get or create global execution monitor instance."""
    global _execution_monitor
    if _execution_monitor is None:
        _execution_monitor = ExecutionMonitor()
    return _execution_monitor


def analyze_execution_result(execution_result: Dict[str, Any], **kwargs) -> ExecutionMetrics:
    """
    Convenient wrapper for execution analysis.
    
    Args:
        execution_result: Result from command execution
        **kwargs: Additional arguments for analyze_execution
        
    Returns:
        ExecutionMetrics with analysis results
    """
    monitor = get_execution_monitor()
    return monitor.analyze_execution(execution_result, **kwargs)