#!/usr/bin/env python3
"""
SelfHealer - Neural Reflection & Self-Repair Agent for MeRNSTA

Analyzes system's own internal architecture, detects weak points, generates
self-repair goals, and routes them to planning and evolution agents.
Enables autonomous architecture self-upgrade and continuous improvement.
"""

import ast
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
import traceback
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
import json

from .base import BaseAgent

@dataclass
class Issue:
    """Represents a detected system issue"""
    issue_id: str
    category: str          # code_quality, architecture, performance, reliability
    severity: str          # critical, high, medium, low
    component: str         # affected component/module
    description: str       # human-readable description
    evidence: List[str]    # supporting evidence (file paths, error messages, etc.)
    impact_score: float    # 0.0 to 1.0
    fix_difficulty: float  # 0.0 to 1.0 (0 = easy, 1 = very hard)
    detected_at: str       # ISO timestamp
    repair_suggestions: List[str] = None
    
    def __post_init__(self):
        if self.repair_suggestions is None:
            self.repair_suggestions = []

@dataclass
class Pattern:
    """Represents an architectural pattern or anti-pattern"""
    pattern_id: str
    pattern_type: str      # anti_pattern, design_smell, performance_issue
    name: str              # human-readable pattern name
    description: str       # what this pattern represents
    locations: List[str]   # where pattern was found
    frequency: int         # how often pattern appears
    risk_level: str        # low, medium, high, critical
    recommended_action: str # what should be done about it

@dataclass
class DiagnosticReport:
    """Comprehensive system diagnostic report"""
    report_id: str
    generated_at: str
    system_health_score: float  # Overall health 0.0 to 1.0
    issues: List[Issue]
    patterns: List[Pattern]
    metrics: Dict[str, Any]     # Various system metrics
    recommendations: List[str]   # Top-level recommendations
    repair_goals: List[str]     # Generated repair goals

class SelfHealer(BaseAgent):
    """
    Self-healing agent that analyzes system architecture and generates repair goals.
    
    Capabilities:
    - Code quality analysis and fragile pattern detection
    - Architecture flaw identification
    - Test coverage and reliability assessment
    - Performance bottleneck detection
    - Autonomous repair goal generation
    - Integration with planning and evolution systems
    """
    
    def __init__(self):
        super().__init__("self_healer")
        
        # Configuration from config.yaml patterns [[memory:4199483]]
        self.analysis_depth = self.agent_config.get('analysis_depth', 'medium')
        self.scan_patterns = self._load_scan_patterns()
        self.severity_thresholds = self._load_severity_thresholds()
        self.repair_templates = self._load_repair_templates()
        
        # System paths and exclusions
        self.project_root = Path.cwd()
        self.exclude_patterns = self.agent_config.get('exclude_patterns', [
            '__pycache__', '.git', '.pytest_cache', 'node_modules', '.venv',
            '*.pyc', '*.pyo', '*.so', '.DS_Store'
        ])
        
        # Initialize connections to other systems
        self._repair_log = None
        self._recursive_planner = None
        self._memory_system = None
        
        # Diagnostic cache
        self._last_diagnostic = None
        self._diagnostic_cache_duration = self.agent_config.get('cache_duration', 300)  # 5 minutes
        
        logging.info(f"[{self.name}] Initialized with {len(self.scan_patterns)} scan patterns")
    
    def _load_scan_patterns(self) -> Dict[str, Any]:
        """Load configurable scan patterns [[memory:4199483]]"""
        default_patterns = {
            'fragile_code': {
                'bare_except': r'except\s*:',
                'global_variables': r'global\s+\w+',
                'hardcoded_paths': r'["\'][/\\][\w/\\.-]+["\']',
                'magic_numbers': r'\b(?<![\w.])\d{2,}\b(?![\w.])',
                'deep_nesting': r'(\s{4,}){4,}',  # 4+ levels of indentation
                'long_lines': r'.{120,}',  # Lines over 120 characters
                'todo_fixme': r'#\s*(TODO|FIXME|HACK|BUG)',
                'duplicate_code': None,  # Handled separately
            },
            'architecture_flaws': {
                'circular_imports': None,  # Detected programmatically
                'god_classes': None,       # Classes with too many methods
                'feature_envy': None,      # Methods using other classes heavily
                'long_parameter_lists': r'def\s+\w+\([^)]{100,}\)',
                'deep_inheritance': None,  # Detected via AST
                'tight_coupling': None,    # High import dependencies
            },
            'performance_issues': {
                'nested_loops': r'for\s+.*:\s*.*for\s+.*:',
                'inefficient_patterns': r'(\.append\(.*\)\s*){3,}',  # Multiple appends
                'memory_leaks': r'global\s+.*=.*\[\]',
                'blocking_calls': r'(time\.sleep|requests\.get|input\()',
            },
            'reliability_issues': {
                'missing_error_handling': None,  # Functions without try/catch
                'resource_leaks': r'open\([^)]*\)(?!\s*with)',
                'race_conditions': r'(threading|multiprocessing).*(?!with.*lock)',
                'assertion_usage': r'assert\s+',  # Asserts in production code
            }
        }
        
        # Get patterns from config, merge with defaults [[memory:4199483]]
        config_patterns = self.agent_config.get('scan_patterns', {})
        return {**default_patterns, **config_patterns}
    
    def _load_severity_thresholds(self) -> Dict[str, Any]:
        """Load severity classification thresholds [[memory:4199483]]"""
        default_thresholds = {
            'critical': {'impact_min': 0.8, 'frequency_min': 10},
            'high': {'impact_min': 0.6, 'frequency_min': 5},
            'medium': {'impact_min': 0.4, 'frequency_min': 2},
            'low': {'impact_min': 0.1, 'frequency_min': 1}
        }
        
        config_thresholds = self.agent_config.get('severity_thresholds', {})
        return {**default_thresholds, **config_thresholds}
    
    def _load_repair_templates(self) -> Dict[str, Any]:
        """Load repair goal templates [[memory:4199483]]"""
        default_templates = {
            'code_quality': [
                "Refactor {component} to eliminate {issue_count} code quality issues",
                "Improve error handling in {component} to increase reliability",
                "Reduce complexity in {component} by breaking down large functions",
                "Add comprehensive documentation to {component}"
            ],
            'architecture': [
                "Redesign {component} to follow better architectural patterns",
                "Break down {component} to reduce coupling and improve modularity",
                "Implement proper interfaces in {component} to reduce dependencies",
                "Refactor {component} to eliminate circular dependencies"
            ],
            'performance': [
                "Optimize {component} to improve performance by {target_improvement}%",
                "Implement caching in {component} to reduce computational overhead",
                "Refactor {component} to use more efficient algorithms",
                "Add performance monitoring to {component}"
            ],
            'reliability': [
                "Add comprehensive error handling to {component}",
                "Implement proper resource management in {component}",
                "Add validation and safety checks to {component}",
                "Increase test coverage for {component} to {target_coverage}%"
            ]
        }
        
        config_templates = self.agent_config.get('repair_templates', {})
        return {**default_templates, **config_templates}
    
    @property
    def repair_log(self):
        """Lazy-load repair log"""
        if self._repair_log is None:
            try:
                from storage.self_repair_log import SelfRepairLog
                self._repair_log = SelfRepairLog()
            except ImportError:
                logging.warning(f"[{self.name}] SelfRepairLog not available")
        return self._repair_log
    
    @property
    def recursive_planner(self):
        """Lazy-load recursive planner"""
        if self._recursive_planner is None:
            try:
                from .recursive_planner import RecursivePlanner
                self._recursive_planner = RecursivePlanner()
            except ImportError:
                logging.warning(f"[{self.name}] RecursivePlanner not available")
        return self._recursive_planner
    
    @property
    def memory_system(self):
        """Lazy-load memory system"""
        if self._memory_system is None:
            try:
                from storage.enhanced_memory_system import EnhancedMemorySystem
                self._memory_system = EnhancedMemorySystem()
            except ImportError:
                try:
                    from storage.memory_log import MemoryLog
                    self._memory_system = MemoryLog()
                except ImportError:
                    logging.warning(f"[{self.name}] Memory system not available")
        return self._memory_system
    
    def get_agent_instructions(self) -> str:
        """Return specialized instructions for the self-healer agent."""
        return (
            "You are a self-healing and architecture analysis specialist. Your role is to "
            "analyze the system's own codebase, detect architectural flaws, identify fragile "
            "patterns, and generate actionable repair goals. Focus on code quality, performance, "
            "reliability, and maintainability. Generate specific, measurable improvement goals "
            "that can be executed by the planning system. "
            f"Analysis depth: {self.analysis_depth}. "
            f"Scanning {len(self.scan_patterns)} pattern categories."
        )
    
    def analyze_code_health(self, target_paths: Optional[List[str]] = None) -> List[Issue]:
        """
        Analyze code health across the system to identify issues.
        
        Args:
            target_paths: Specific paths to analyze, or None for full system scan
            
        Returns:
            List of detected issues with severity and repair suggestions
        """
        if not self.enabled:
            return []
        
        try:
            logging.info(f"[{self.name}] Starting code health analysis")
            
            # Determine scan scope
            if target_paths:
                scan_paths = [Path(p) for p in target_paths]
            else:
                scan_paths = [self.project_root]
            
            issues = []
            
            # Scan for fragile code patterns
            issues.extend(self._scan_fragile_patterns(scan_paths))
            
            # Analyze test coverage
            issues.extend(self._analyze_test_coverage(scan_paths))
            
            # Check for volatile modules
            issues.extend(self._detect_volatile_modules(scan_paths))
            
            # Analyze error handling
            issues.extend(self._analyze_error_handling(scan_paths))
            
            # Check for performance issues
            issues.extend(self._scan_performance_issues(scan_paths))
            
            # Analyze complexity metrics
            issues.extend(self._analyze_complexity(scan_paths))
            
            # Check for security issues
            issues.extend(self._scan_security_issues(scan_paths))
            
            logging.info(f"[{self.name}] Code health analysis found {len(issues)} issues")
            
            return issues
            
        except Exception as e:
            logging.error(f"[{self.name}] Error in code health analysis: {e}")
            return []
    
    def detect_architecture_flaws(self, target_paths: Optional[List[str]] = None) -> List[Pattern]:
        """
        Detect architectural patterns and anti-patterns.
        
        Args:
            target_paths: Specific paths to analyze
            
        Returns:
            List of detected architectural patterns
        """
        if not self.enabled:
            return []
        
        try:
            logging.info(f"[{self.name}] Starting architecture analysis")
            
            patterns = []
            
            # Detect circular imports
            patterns.extend(self._detect_circular_imports())
            
            # Find god classes
            patterns.extend(self._detect_god_classes())
            
            # Identify tight coupling
            patterns.extend(self._detect_tight_coupling())
            
            # Check inheritance depth
            patterns.extend(self._analyze_inheritance_depth())
            
            # Detect feature envy
            patterns.extend(self._detect_feature_envy())
            
            # Analyze module cohesion
            patterns.extend(self._analyze_module_cohesion())
            
            logging.info(f"[{self.name}] Architecture analysis found {len(patterns)} patterns")
            
            return patterns
            
        except Exception as e:
            logging.error(f"[{self.name}] Error in architecture analysis: {e}")
            return []
    
    def generate_repair_goals(self, issues: List[Issue], patterns: Optional[List[Pattern]] = None) -> List[str]:
        """
        Generate actionable repair goals based on detected issues.
        
        Args:
            issues: List of detected issues
            patterns: Optional list of architectural patterns
            
        Returns:
            List of specific repair goal descriptions
        """
        if not issues:
            return []
        
        try:
            repair_goals = []
            patterns = patterns or []
            
            # Group issues by category and component
            issues_by_category = defaultdict(list)
            issues_by_component = defaultdict(list)
            
            for issue in issues:
                issues_by_category[issue.category].append(issue)
                issues_by_component[issue.component].append(issue)
            
            # Generate category-based repair goals
            for category, category_issues in issues_by_category.items():
                high_impact_issues = [i for i in category_issues if i.impact_score >= 0.6]
                
                if high_impact_issues:
                    templates = self.repair_templates.get(category, [])
                    if templates:
                        template = templates[0]  # Use first template for category
                        
                        # Find most affected component
                        component_counts = Counter(i.component for i in high_impact_issues)
                        top_component = component_counts.most_common(1)[0][0]
                        
                        goal = template.format(
                            component=top_component,
                            issue_count=len(high_impact_issues),
                            target_improvement=20,  # Default improvement target
                            target_coverage=80      # Default coverage target
                        )
                        repair_goals.append(goal)
            
            # Generate component-specific goals for heavily affected components
            for component, component_issues in issues_by_component.items():
                if len(component_issues) >= 3:  # Component with multiple issues
                    severity_counts = Counter(i.severity for i in component_issues)
                    
                    if severity_counts.get('critical', 0) > 0 or severity_counts.get('high', 0) >= 2:
                        goal = f"Comprehensively refactor {component} to address {len(component_issues)} identified issues"
                        repair_goals.append(goal)
            
            # Generate pattern-based goals
            for pattern in patterns:
                if pattern.risk_level in ['high', 'critical'] and pattern.frequency >= 3:
                    goal = f"Address {pattern.name} anti-pattern found in {pattern.frequency} locations"
                    repair_goals.append(goal)
            
            # Generate meta-goals for systemic issues
            total_critical = len([i for i in issues if i.severity == 'critical'])
            total_high = len([i for i in issues if i.severity == 'high'])
            
            if total_critical >= 5:
                repair_goals.append(f"Implement systematic approach to resolve {total_critical} critical system issues")
            
            if total_high >= 10:
                repair_goals.append(f"Establish architecture review process to prevent accumulation of {total_high} high-severity issues")
            
            # Add monitoring and prevention goals
            if len(issues) >= 20:
                repair_goals.extend([
                    "Implement automated code quality monitoring to catch issues early",
                    "Establish code review standards to prevent architecture degradation",
                    "Set up continuous integration checks for code quality metrics"
                ])
            
            logging.info(f"[{self.name}] Generated {len(repair_goals)} repair goals from {len(issues)} issues")
            
            return repair_goals
            
        except Exception as e:
            logging.error(f"[{self.name}] Error generating repair goals: {e}")
            return []
    
    def prioritize_repairs(self, goals: List[str]) -> List[str]:
        """
        Prioritize repair goals by impact, urgency, and feasibility.
        
        Args:
            goals: List of repair goal descriptions
            
        Returns:
            Goals sorted by priority (highest first)
        """
        if not goals:
            return []
        
        try:
            # Score each goal
            goal_scores = []
            
            for goal in goals:
                score = self._score_repair_goal(goal)
                goal_scores.append((goal, score))
            
            # Sort by score (highest first)
            goal_scores.sort(key=lambda x: x[1], reverse=True)
            
            prioritized = [goal for goal, score in goal_scores]
            
            logging.info(f"[{self.name}] Prioritized {len(goals)} repair goals")
            
            return prioritized
            
        except Exception as e:
            logging.error(f"[{self.name}] Error prioritizing repairs: {e}")
            return goals  # Return original order on error
    
    def run_diagnostic_suite(self) -> DiagnosticReport:
        """
        Run comprehensive diagnostic suite across the entire system.
        
        Returns:
            Complete diagnostic report with issues, patterns, and recommendations
        """
        if not self.enabled:
            return DiagnosticReport("", "", 0.0, [], [], {}, [], [])
        
        # Check cache first
        if (self._last_diagnostic and 
            datetime.fromisoformat(self._last_diagnostic.generated_at) > 
            datetime.now() - timedelta(seconds=self._diagnostic_cache_duration)):
            logging.info(f"[{self.name}] Returning cached diagnostic report")
            return self._last_diagnostic
        
        try:
            start_time = time.time()
            logging.info(f"[{self.name}] Starting comprehensive diagnostic suite")
            
            # Analyze code health
            issues = self.analyze_code_health()
            
            # Detect architecture flaws
            patterns = self.detect_architecture_flaws()
            
            # Generate repair goals
            repair_goals = self.generate_repair_goals(issues, patterns)
            prioritized_goals = self.prioritize_repairs(repair_goals)
            
            # Calculate system metrics
            metrics = self._calculate_system_metrics(issues, patterns)
            
            # Calculate overall health score
            health_score = self._calculate_health_score(issues, patterns, metrics)
            
            # Generate top-level recommendations
            recommendations = self._generate_recommendations(issues, patterns, metrics)
            
            # Create report
            report = DiagnosticReport(
                report_id=f"diagnostic-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                generated_at=datetime.now().isoformat(),
                system_health_score=health_score,
                issues=issues,
                patterns=patterns,
                metrics=metrics,
                recommendations=recommendations,
                repair_goals=prioritized_goals
            )
            
            # Cache the report
            self._last_diagnostic = report
            
            duration = time.time() - start_time
            logging.info(f"[{self.name}] Diagnostic suite completed in {duration:.2f}s - Health Score: {health_score:.2f}")
            
            return report
            
        except Exception as e:
            logging.error(f"[{self.name}] Error in diagnostic suite: {e}")
            # Return empty report on error
            return DiagnosticReport(
                report_id=f"error-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                generated_at=datetime.now().isoformat(),
                system_health_score=0.0,
                issues=[],
                patterns=[],
                metrics={"error": str(e)},
                recommendations=["Investigate diagnostic system failure"],
                repair_goals=[]
            )
    
    def _scan_fragile_patterns(self, scan_paths: List[Path]) -> List[Issue]:
        """Scan for fragile code patterns"""
        issues = []
        
        for path in scan_paths:
            for py_file in self._get_python_files(path):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    issues.extend(self._analyze_file_patterns(py_file, content))
                except Exception as e:
                    logging.warning(f"[{self.name}] Could not analyze {py_file}: {e}")
        
        return issues
    
    def _analyze_file_patterns(self, file_path: Path, content: str) -> List[Issue]:
        """Analyze a single file for fragile patterns"""
        issues = []
        lines = content.split('\n')
        
        fragile_patterns = self.scan_patterns.get('fragile_code', {})
        
        for pattern_name, pattern_regex in fragile_patterns.items():
            if pattern_regex is None:
                continue
                
            matches = re.finditer(pattern_regex, content, re.MULTILINE)
            
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                
                issue = Issue(
                    issue_id=f"{pattern_name}-{file_path.name}-{line_num}",
                    category="code_quality",
                    severity=self._classify_pattern_severity(pattern_name),
                    component=str(file_path.relative_to(self.project_root)),
                    description=f"{pattern_name.replace('_', ' ').title()} detected",
                    evidence=[f"{file_path}:{line_num} - {lines[line_num-1].strip()}" if line_num <= len(lines) else f"{file_path}:{line_num}"],
                    impact_score=self._calculate_pattern_impact(pattern_name),
                    fix_difficulty=self._estimate_fix_difficulty(pattern_name),
                    detected_at=datetime.now().isoformat(),
                    repair_suggestions=self._get_pattern_suggestions(pattern_name)
                )
                issues.append(issue)
        
        return issues
    
    def _analyze_test_coverage(self, scan_paths: List[Path]) -> List[Issue]:
        """Analyze test coverage and identify low-coverage areas"""
        issues = []
        
        try:
            # Find test files and production files
            test_files = set()
            prod_files = set()
            
            for path in scan_paths:
                for py_file in self._get_python_files(path):
                    if 'test' in py_file.name.lower() or py_file.parent.name == 'tests':
                        test_files.add(py_file)
                    else:
                        prod_files.add(py_file)
            
            # Calculate coverage ratio
            if prod_files:
                coverage_ratio = len(test_files) / len(prod_files)
                
                if coverage_ratio < 0.3:  # Less than 30% test coverage
                    issue = Issue(
                        issue_id="low-test-coverage",
                        category="reliability",
                        severity="high" if coverage_ratio < 0.1 else "medium",
                        component="test_suite",
                        description=f"Low test coverage: {coverage_ratio:.1%}",
                        evidence=[f"{len(test_files)} test files for {len(prod_files)} production files"],
                        impact_score=0.8 if coverage_ratio < 0.1 else 0.6,
                        fix_difficulty=0.7,
                        detected_at=datetime.now().isoformat(),
                        repair_suggestions=[
                            "Add unit tests for critical functions",
                            "Implement integration tests for key workflows",
                            "Set up automated coverage reporting"
                        ]
                    )
                    issues.append(issue)
        
        except Exception as e:
            logging.warning(f"[{self.name}] Error analyzing test coverage: {e}")
        
        return issues
    
    def _detect_volatile_modules(self, scan_paths: List[Path]) -> List[Issue]:
        """Detect modules that change frequently (potential instability)"""
        issues = []
        
        try:
            # This would integrate with git history or change tracking
            # For now, detect based on TODO/FIXME comments and complexity
            
            for path in scan_paths:
                for py_file in self._get_python_files(path):
                    content = py_file.read_text(encoding='utf-8')
                    
                    # Count TODO/FIXME comments as volatility indicators
                    todo_count = len(re.findall(r'#\s*(TODO|FIXME|HACK|BUG)', content, re.IGNORECASE))
                    
                    # Count function definitions as complexity indicator
                    func_count = len(re.findall(r'def\s+\w+', content))
                    
                    # Calculate volatility score
                    volatility_score = (todo_count * 0.3 + func_count * 0.1) / max(1, len(content.split('\n')) / 100)
                    
                    if volatility_score > 0.5:  # High volatility threshold
                        issue = Issue(
                            issue_id=f"volatile-module-{py_file.stem}",
                            category="architecture",
                            severity="medium" if volatility_score < 0.8 else "high",
                            component=str(py_file.relative_to(self.project_root)),
                            description=f"High volatility module (score: {volatility_score:.2f})",
                            evidence=[f"{todo_count} TODO/FIXME comments, {func_count} functions"],
                            impact_score=min(0.9, volatility_score),
                            fix_difficulty=0.6,
                            detected_at=datetime.now().isoformat(),
                            repair_suggestions=[
                                "Refactor to reduce complexity",
                                "Address TODO/FIXME comments",
                                "Split large modules into smaller components"
                            ]
                        )
                        issues.append(issue)
        
        except Exception as e:
            logging.warning(f"[{self.name}] Error detecting volatile modules: {e}")
        
        return issues
    
    def _analyze_error_handling(self, scan_paths: List[Path]) -> List[Issue]:
        """Analyze error handling patterns"""
        issues = []
        
        try:
            for path in scan_paths:
                for py_file in self._get_python_files(path):
                    content = py_file.read_text(encoding='utf-8')
                    
                    # Find functions without error handling
                    functions = re.findall(r'def\s+(\w+)\s*\([^)]*\):', content)
                    try_blocks = len(re.findall(r'\btry\s*:', content))
                    
                    if len(functions) > 3 and try_blocks == 0:
                        issue = Issue(
                            issue_id=f"no-error-handling-{py_file.stem}",
                            category="reliability",
                            severity="medium",
                            component=str(py_file.relative_to(self.project_root)),
                            description=f"No error handling in {len(functions)} functions",
                            evidence=[f"{len(functions)} functions, 0 try blocks"],
                            impact_score=0.6,
                            fix_difficulty=0.4,
                            detected_at=datetime.now().isoformat(),
                            repair_suggestions=[
                                "Add try-except blocks for external calls",
                                "Implement input validation",
                                "Add logging for error cases"
                            ]
                        )
                        issues.append(issue)
        
        except Exception as e:
            logging.warning(f"[{self.name}] Error analyzing error handling: {e}")
        
        return issues
    
    def _scan_performance_issues(self, scan_paths: List[Path]) -> List[Issue]:
        """Scan for performance anti-patterns"""
        issues = []
        
        performance_patterns = self.scan_patterns.get('performance_issues', {})
        
        for path in scan_paths:
            for py_file in self._get_python_files(path):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    
                    for pattern_name, pattern_regex in performance_patterns.items():
                        if pattern_regex is None:
                            continue
                            
                        matches = list(re.finditer(pattern_regex, content))
                        
                        if matches:
                            issue = Issue(
                                issue_id=f"{pattern_name}-{py_file.stem}",
                                category="performance",
                                severity="medium",
                                component=str(py_file.relative_to(self.project_root)),
                                description=f"Performance issue: {pattern_name.replace('_', ' ')}",
                                evidence=[f"{len(matches)} occurrences found"],
                                impact_score=0.5,
                                fix_difficulty=0.5,
                                detected_at=datetime.now().isoformat(),
                                repair_suggestions=self._get_performance_suggestions(pattern_name)
                            )
                            issues.append(issue)
                
                except Exception as e:
                    logging.warning(f"[{self.name}] Could not scan {py_file} for performance issues: {e}")
        
        return issues
    
    def _analyze_complexity(self, scan_paths: List[Path]) -> List[Issue]:
        """Analyze code complexity metrics"""
        issues = []
        
        try:
            for path in scan_paths:
                for py_file in self._get_python_files(path):
                    content = py_file.read_text(encoding='utf-8')
                    
                    # Simple complexity metrics
                    lines = content.split('\n')
                    non_empty_lines = [l for l in lines if l.strip()]
                    
                    # Check for very long files
                    if len(non_empty_lines) > 500:
                        issue = Issue(
                            issue_id=f"long-file-{py_file.stem}",
                            category="architecture",
                            severity="medium",
                            component=str(py_file.relative_to(self.project_root)),
                            description=f"Very long file ({len(non_empty_lines)} lines)",
                            evidence=[f"{len(non_empty_lines)} non-empty lines"],
                            impact_score=0.5,
                            fix_difficulty=0.8,
                            detected_at=datetime.now().isoformat(),
                            repair_suggestions=[
                                "Split file into smaller modules",
                                "Extract classes to separate files",
                                "Group related functionality"
                            ]
                        )
                        issues.append(issue)
                    
                    # Check for very long functions
                    long_functions = re.findall(r'def\s+(\w+).*?(?=def|\Z)', content, re.DOTALL)
                    for func in long_functions:
                        func_lines = func.split('\n')
                        if len([l for l in func_lines if l.strip()]) > 50:
                            func_name = func.split('(')[0].replace('def ', '').strip()
                            issue = Issue(
                                issue_id=f"long-function-{py_file.stem}-{func_name}",
                                category="code_quality",
                                severity="low",
                                component=str(py_file.relative_to(self.project_root)),
                                description=f"Long function: {func_name}",
                                evidence=[f"Function has {len(func_lines)} lines"],
                                impact_score=0.3,
                                fix_difficulty=0.6,
                                detected_at=datetime.now().isoformat(),
                                repair_suggestions=[
                                    "Break function into smaller functions",
                                    "Extract helper methods",
                                    "Use composition instead of long procedures"
                                ]
                            )
                            issues.append(issue)
        
        except Exception as e:
            logging.warning(f"[{self.name}] Error analyzing complexity: {e}")
        
        return issues
    
    def _scan_security_issues(self, scan_paths: List[Path]) -> List[Issue]:
        """Scan for potential security issues"""
        issues = []
        
        security_patterns = {
            'hardcoded_secrets': r'(password|secret|key|token)\s*=\s*["\'][\w\-/+=]{8,}["\']',
            'sql_injection': r'(execute|cursor|query).*%.*\+',
            'command_injection': r'(os\.system|subprocess\.call).*\+',
            'unsafe_eval': r'\beval\s*\(',
            'unsafe_exec': r'\bexec\s*\(',
        }
        
        for path in scan_paths:
            for py_file in self._get_python_files(path):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    
                    for pattern_name, pattern_regex in security_patterns.items():
                        matches = list(re.finditer(pattern_regex, content, re.IGNORECASE))
                        
                        if matches:
                            issue = Issue(
                                issue_id=f"{pattern_name}-{py_file.stem}",
                                category="security",
                                severity="high" if 'injection' in pattern_name else "medium",
                                component=str(py_file.relative_to(self.project_root)),
                                description=f"Security issue: {pattern_name.replace('_', ' ')}",
                                evidence=[f"{len(matches)} potential vulnerabilities"],
                                impact_score=0.8 if 'injection' in pattern_name else 0.6,
                                fix_difficulty=0.5,
                                detected_at=datetime.now().isoformat(),
                                repair_suggestions=self._get_security_suggestions(pattern_name)
                            )
                            issues.append(issue)
                
                except Exception as e:
                    logging.warning(f"[{self.name}] Could not scan {py_file} for security issues: {e}")
        
        return issues
    
    def _detect_circular_imports(self) -> List[Pattern]:
        """Detect circular import dependencies"""
        patterns = []
        
        try:
            import_graph = defaultdict(set)
            
            # Build import dependency graph
            for py_file in self._get_python_files(self.project_root):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    
                    # Find import statements
                    imports = re.findall(r'from\s+([\w.]+)\s+import|import\s+([\w.]+)', content)
                    
                    current_module = str(py_file.relative_to(self.project_root)).replace('/', '.').replace('.py', '')
                    
                    for from_import, direct_import in imports:
                        imported = from_import or direct_import
                        if imported and imported.startswith('.'):  # Relative import
                            import_graph[current_module].add(imported)
                
                except Exception:
                    continue
            
            # Detect cycles (simplified cycle detection)
            cycles_found = []
            
            for module, imports in import_graph.items():
                for imported in imports:
                    if module in import_graph.get(imported, set()):
                        cycles_found.append((module, imported))
            
            if cycles_found:
                pattern = Pattern(
                    pattern_id="circular-imports",
                    pattern_type="anti_pattern",
                    name="Circular Import Dependencies",
                    description="Modules that import each other creating circular dependencies",
                    locations=[f"{a} <-> {b}" for a, b in cycles_found],
                    frequency=len(cycles_found),
                    risk_level="high" if len(cycles_found) > 2 else "medium",
                    recommended_action="Refactor to break circular dependencies using interfaces or dependency injection"
                )
                patterns.append(pattern)
        
        except Exception as e:
            logging.warning(f"[{self.name}] Error detecting circular imports: {e}")
        
        return patterns
    
    def _detect_god_classes(self) -> List[Pattern]:
        """Detect classes with too many methods (god classes)"""
        patterns = []
        
        try:
            god_classes = []
            
            for py_file in self._get_python_files(self.project_root):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                            
                            if len(methods) > 15:  # Threshold for god class
                                god_classes.append(f"{py_file.stem}.{node.name} ({len(methods)} methods)")
                
                except Exception:
                    continue
            
            if god_classes:
                pattern = Pattern(
                    pattern_id="god-classes",
                    pattern_type="anti_pattern",
                    name="God Classes",
                    description="Classes with excessive responsibilities (too many methods)",
                    locations=god_classes,
                    frequency=len(god_classes),
                    risk_level="high" if len(god_classes) > 3 else "medium",
                    recommended_action="Break down large classes using composition or extract method pattern"
                )
                patterns.append(pattern)
        
        except Exception as e:
            logging.warning(f"[{self.name}] Error detecting god classes: {e}")
        
        return patterns
    
    def _detect_tight_coupling(self) -> List[Pattern]:
        """Detect tight coupling between modules"""
        patterns = []
        
        try:
            # Count imports between modules
            coupling_matrix = defaultdict(lambda: defaultdict(int))
            
            for py_file in self._get_python_files(self.project_root):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    current_module = py_file.stem
                    
                    # Count imports from other local modules
                    local_imports = re.findall(r'from\s+([\w.]+)\s+import|import\s+([\w.]+)', content)
                    
                    for from_import, direct_import in local_imports:
                        imported = (from_import or direct_import).split('.')[0]
                        if imported and not imported.startswith('_') and imported != current_module:
                            coupling_matrix[current_module][imported] += 1
                
                except Exception:
                    continue
            
            # Find highly coupled modules
            high_coupling = []
            for module, imports in coupling_matrix.items():
                total_imports = sum(imports.values())
                if total_imports > 10:  # High coupling threshold
                    high_coupling.append(f"{module} ({total_imports} imports)")
            
            if high_coupling:
                pattern = Pattern(
                    pattern_id="tight-coupling",
                    pattern_type="design_smell",
                    name="Tight Coupling",
                    description="Modules with excessive dependencies on other modules",
                    locations=high_coupling,
                    frequency=len(high_coupling),
                    risk_level="medium",
                    recommended_action="Reduce dependencies through interfaces and dependency injection"
                )
                patterns.append(pattern)
        
        except Exception as e:
            logging.warning(f"[{self.name}] Error detecting tight coupling: {e}")
        
        return patterns
    
    def _analyze_inheritance_depth(self) -> List[Pattern]:
        """Analyze inheritance depth and detect deep hierarchies"""
        patterns = []
        
        try:
            deep_hierarchies = []
            
            for py_file in self._get_python_files(self.project_root):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            if len(node.bases) > 2:  # Multiple inheritance
                                deep_hierarchies.append(f"{py_file.stem}.{node.name} (multiple inheritance)")
                
                except Exception:
                    continue
            
            if deep_hierarchies:
                pattern = Pattern(
                    pattern_id="deep-inheritance",
                    pattern_type="design_smell",
                    name="Deep Inheritance Hierarchies",
                    description="Classes with complex inheritance patterns",
                    locations=deep_hierarchies,
                    frequency=len(deep_hierarchies),
                    risk_level="medium",
                    recommended_action="Prefer composition over inheritance for complex relationships"
                )
                patterns.append(pattern)
        
        except Exception as e:
            logging.warning(f"[{self.name}] Error analyzing inheritance: {e}")
        
        return patterns
    
    def _detect_feature_envy(self) -> List[Pattern]:
        """Detect feature envy (methods using other classes heavily)"""
        patterns = []
        
        try:
            # This is a simplified detection - would need more sophisticated analysis
            feature_envy_cases = []
            
            for py_file in self._get_python_files(self.project_root):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    
                    # Look for methods with many external calls
                    external_calls = re.findall(r'(\w+)\.(\w+)\(', content)
                    
                    if len(external_calls) > 20:  # Many external calls
                        feature_envy_cases.append(f"{py_file.stem} ({len(external_calls)} external calls)")
                
                except Exception:
                    continue
            
            if feature_envy_cases:
                pattern = Pattern(
                    pattern_id="feature-envy",
                    pattern_type="design_smell",
                    name="Feature Envy",
                    description="Classes that use other classes' methods excessively",
                    locations=feature_envy_cases,
                    frequency=len(feature_envy_cases),
                    risk_level="low",
                    recommended_action="Move methods closer to the data they operate on"
                )
                patterns.append(pattern)
        
        except Exception as e:
            logging.warning(f"[{self.name}] Error detecting feature envy: {e}")
        
        return patterns
    
    def _analyze_module_cohesion(self) -> List[Pattern]:
        """Analyze module cohesion and identify low-cohesion modules"""
        patterns = []
        
        try:
            low_cohesion_modules = []
            
            for py_file in self._get_python_files(self.project_root):
                try:
                    content = py_file.read_text(encoding='utf-8')
                    
                    # Simple cohesion analysis: functions vs classes ratio
                    functions = len(re.findall(r'def\s+\w+', content))
                    classes = len(re.findall(r'class\s+\w+', content))
                    
                    # If many functions and few classes, might indicate low cohesion
                    if functions > 15 and classes < 2:
                        low_cohesion_modules.append(f"{py_file.stem} ({functions} functions, {classes} classes)")
                
                except Exception:
                    continue
            
            if low_cohesion_modules:
                pattern = Pattern(
                    pattern_id="low-cohesion",
                    pattern_type="design_smell",
                    name="Low Module Cohesion",
                    description="Modules with many unrelated functions",
                    locations=low_cohesion_modules,
                    frequency=len(low_cohesion_modules),
                    risk_level="low",
                    recommended_action="Group related functions into classes or separate modules"
                )
                patterns.append(pattern)
        
        except Exception as e:
            logging.warning(f"[{self.name}] Error analyzing cohesion: {e}")
        
        return patterns
    
    def _get_python_files(self, path: Path) -> List[Path]:
        """Get all Python files in a path, respecting exclusions"""
        python_files = []
        
        if path.is_file() and path.suffix == '.py':
            return [path]
        
        try:
            for py_file in path.rglob('*.py'):
                # Check exclusion patterns
                if any(pattern in str(py_file) for pattern in self.exclude_patterns):
                    continue
                
                python_files.append(py_file)
        
        except Exception as e:
            logging.warning(f"[{self.name}] Error scanning {path}: {e}")
        
        return python_files
    
    def _classify_pattern_severity(self, pattern_name: str) -> str:
        """Classify pattern severity based on pattern type"""
        severity_map = {
            'bare_except': 'high',
            'global_variables': 'medium',
            'hardcoded_paths': 'medium',
            'magic_numbers': 'low',
            'deep_nesting': 'medium',
            'long_lines': 'low',
            'todo_fixme': 'low'
        }
        
        return severity_map.get(pattern_name, 'medium')
    
    def _calculate_pattern_impact(self, pattern_name: str) -> float:
        """Calculate impact score for a pattern"""
        impact_map = {
            'bare_except': 0.8,
            'global_variables': 0.6,
            'hardcoded_paths': 0.7,
            'magic_numbers': 0.3,
            'deep_nesting': 0.5,
            'long_lines': 0.2,
            'todo_fixme': 0.3
        }
        
        return impact_map.get(pattern_name, 0.5)
    
    def _estimate_fix_difficulty(self, pattern_name: str) -> float:
        """Estimate difficulty of fixing a pattern"""
        difficulty_map = {
            'bare_except': 0.4,
            'global_variables': 0.7,
            'hardcoded_paths': 0.5,
            'magic_numbers': 0.3,
            'deep_nesting': 0.6,
            'long_lines': 0.2,
            'todo_fixme': 0.4
        }
        
        return difficulty_map.get(pattern_name, 0.5)
    
    def _get_pattern_suggestions(self, pattern_name: str) -> List[str]:
        """Get repair suggestions for a pattern"""
        suggestions_map = {
            'bare_except': [
                "Replace bare except with specific exception types",
                "Add logging to exception handlers",
                "Consider if exception should be re-raised"
            ],
            'global_variables': [
                "Encapsulate global state in classes",
                "Use dependency injection",
                "Convert to module-level constants if read-only"
            ],
            'hardcoded_paths': [
                "Use configuration files for paths",
                "Use environment variables",
                "Use pathlib for cross-platform paths"
            ],
            'magic_numbers': [
                "Define named constants",
                "Extract to configuration",
                "Add comments explaining the value"
            ],
            'deep_nesting': [
                "Extract nested logic to functions",
                "Use early returns to reduce nesting",
                "Consider guard clauses"
            ],
            'long_lines': [
                "Break long lines at logical points",
                "Extract complex expressions to variables",
                "Use line continuation for readability"
            ],
            'todo_fixme': [
                "Address outstanding TODO items",
                "Create tickets for significant work",
                "Remove obsolete comments"
            ]
        }
        
        return suggestions_map.get(pattern_name, ["Review and refactor as needed"])
    
    def _get_performance_suggestions(self, pattern_name: str) -> List[str]:
        """Get performance improvement suggestions"""
        suggestions_map = {
            'nested_loops': [
                "Consider algorithmic optimization",
                "Use more efficient data structures",
                "Cache intermediate results"
            ],
            'inefficient_patterns': [
                "Use list comprehensions",
                "Batch operations where possible",
                "Consider using numpy for numerical operations"
            ],
            'memory_leaks': [
                "Review global variable usage",
                "Implement proper cleanup",
                "Use context managers for resources"
            ],
            'blocking_calls': [
                "Use asynchronous operations",
                "Implement non-blocking alternatives",
                "Add timeouts to blocking calls"
            ]
        }
        
        return suggestions_map.get(pattern_name, ["Review performance characteristics"])
    
    def _get_security_suggestions(self, pattern_name: str) -> List[str]:
        """Get security improvement suggestions"""
        suggestions_map = {
            'hardcoded_secrets': [
                "Move secrets to environment variables",
                "Use secure secret management",
                "Rotate exposed secrets immediately"
            ],
            'sql_injection': [
                "Use parameterized queries",
                "Validate and sanitize inputs",
                "Use ORM query builders"
            ],
            'command_injection': [
                "Validate command arguments",
                "Use subprocess with list arguments",
                "Avoid shell=True in subprocess calls"
            ],
            'unsafe_eval': [
                "Remove eval() usage",
                "Use ast.literal_eval() for safe evaluation",
                "Implement proper parsing"
            ],
            'unsafe_exec': [
                "Remove exec() usage",
                "Use proper code loading mechanisms",
                "Implement sandboxing if dynamic execution needed"
            ]
        }
        
        return suggestions_map.get(pattern_name, ["Review security implications"])
    
    def _score_repair_goal(self, goal: str) -> float:
        """Score a repair goal for prioritization"""
        score = 0.5  # Base score
        
        goal_lower = goal.lower()
        
        # High priority keywords
        high_priority = ['critical', 'security', 'performance', 'reliability', 'fix']
        for keyword in high_priority:
            if keyword in goal_lower:
                score += 0.2
        
        # Impact indicators
        impact_indicators = ['system', 'architecture', 'comprehensive', 'multiple']
        for indicator in impact_indicators:
            if indicator in goal_lower:
                score += 0.1
        
        # Urgency indicators
        urgency_indicators = ['urgent', 'immediate', 'failing', 'broken']
        for indicator in urgency_indicators:
            if indicator in goal_lower:
                score += 0.15
        
        # Technical debt indicators
        debt_indicators = ['refactor', 'cleanup', 'improve', 'optimize']
        for indicator in debt_indicators:
            if indicator in goal_lower:
                score += 0.05
        
        return min(1.0, score)
    
    def _calculate_system_metrics(self, issues: List[Issue], patterns: List[Pattern]) -> Dict[str, Any]:
        """Calculate overall system metrics"""
        metrics = {}
        
        # Issue metrics
        metrics['total_issues'] = len(issues)
        metrics['critical_issues'] = len([i for i in issues if i.severity == 'critical'])
        metrics['high_issues'] = len([i for i in issues if i.severity == 'high'])
        metrics['medium_issues'] = len([i for i in issues if i.severity == 'medium'])
        metrics['low_issues'] = len([i for i in issues if i.severity == 'low'])
        
        # Category breakdown
        by_category = defaultdict(int)
        for issue in issues:
            by_category[issue.category] += 1
        metrics['issues_by_category'] = dict(by_category)
        
        # Pattern metrics
        metrics['total_patterns'] = len(patterns)
        metrics['high_risk_patterns'] = len([p for p in patterns if p.risk_level in ['high', 'critical']])
        
        # Component health
        component_issues = defaultdict(int)
        for issue in issues:
            component_issues[issue.component] += 1
        
        metrics['most_problematic_components'] = sorted(
            component_issues.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # Average scores
        if issues:
            metrics['average_impact_score'] = sum(i.impact_score for i in issues) / len(issues)
            metrics['average_fix_difficulty'] = sum(i.fix_difficulty for i in issues) / len(issues)
        else:
            metrics['average_impact_score'] = 0.0
            metrics['average_fix_difficulty'] = 0.0
        
        return metrics
    
    def _calculate_health_score(self, issues: List[Issue], patterns: List[Pattern], metrics: Dict[str, Any]) -> float:
        """Calculate overall system health score (0.0 to 1.0)"""
        
        # Start with perfect health
        health_score = 1.0
        
        # Deduct for issues based on severity
        critical_weight = 0.15
        high_weight = 0.08
        medium_weight = 0.03
        low_weight = 0.01
        
        health_score -= metrics.get('critical_issues', 0) * critical_weight
        health_score -= metrics.get('high_issues', 0) * high_weight
        health_score -= metrics.get('medium_issues', 0) * medium_weight
        health_score -= metrics.get('low_issues', 0) * low_weight
        
        # Deduct for high-risk patterns
        high_risk_patterns = metrics.get('high_risk_patterns', 0)
        health_score -= high_risk_patterns * 0.05
        
        # Ensure score stays within bounds
        return max(0.0, min(1.0, health_score))
    
    def _generate_recommendations(self, issues: List[Issue], patterns: List[Pattern], metrics: Dict[str, Any]) -> List[str]:
        """Generate top-level recommendations"""
        recommendations = []
        
        # Critical issues recommendations
        critical_count = metrics.get('critical_issues', 0)
        if critical_count > 0:
            recommendations.append(f"Address {critical_count} critical issues immediately")
        
        # Most problematic components
        problematic = metrics.get('most_problematic_components', [])
        if problematic:
            top_component, issue_count = problematic[0]
            if issue_count >= 5:
                recommendations.append(f"Focus refactoring efforts on {top_component} ({issue_count} issues)")
        
        # Pattern-based recommendations
        high_risk_patterns = metrics.get('high_risk_patterns', 0)
        if high_risk_patterns > 3:
            recommendations.append("Implement architecture review process to prevent pattern degradation")
        
        # Category-specific recommendations
        issues_by_category = metrics.get('issues_by_category', {})
        
        if issues_by_category.get('security', 0) > 0:
            recommendations.append("Conduct security audit and implement secure coding practices")
        
        if issues_by_category.get('performance', 0) >= 3:
            recommendations.append("Performance optimization needed in multiple components")
        
        if issues_by_category.get('reliability', 0) >= 5:
            recommendations.append("Improve error handling and testing coverage")
        
        # Overall health recommendations
        health_score = self._calculate_health_score(issues, patterns, metrics)
        
        if health_score < 0.5:
            recommendations.append("System health is poor - comprehensive refactoring recommended")
        elif health_score < 0.7:
            recommendations.append("System health is below average - targeted improvements needed")
        
        return recommendations
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """Handle direct messages to the self-healer agent"""
        if not self.enabled:
            return f"[{self.name}] Agent disabled"
        
        try:
            message_lower = message.lower()
            
            if any(phrase in message_lower for phrase in ['diagnose', 'analyze health', 'check system']):
                report = self.run_diagnostic_suite()
                
                response = f"🔍 **System Diagnostic Complete**\n"
                response += f"📊 **Health Score:** {report.system_health_score:.2f}/1.0\n"
                response += f"🐛 **Issues Found:** {len(report.issues)}\n"
                response += f"⚠️ **Patterns Detected:** {len(report.patterns)}\n"
                response += f"🎯 **Repair Goals:** {len(report.repair_goals)}\n\n"
                
                if report.repair_goals:
                    response += "**Top Repair Goals:**\n"
                    for i, goal in enumerate(report.repair_goals[:3], 1):
                        response += f"{i}. {goal}\n"
                
                return response
            
            elif any(phrase in message_lower for phrase in ['repair goals', 'generate repairs']):
                issues = self.analyze_code_health()
                patterns = self.detect_architecture_flaws()
                goals = self.generate_repair_goals(issues, patterns)
                prioritized = self.prioritize_repairs(goals)
                
                response = f"🔧 **Generated {len(prioritized)} Repair Goals:**\n\n"
                for i, goal in enumerate(prioritized, 1):
                    response += f"{i}. {goal}\n"
                
                return response
            
            else:
                return self._provide_self_healer_guidance(message, context)
        
        except Exception as e:
            logging.error(f"[{self.name}] Error in respond: {e}")
            return f"[{self.name}] Error: {str(e)}"
    
    def _provide_self_healer_guidance(self, message: str, context: Optional[Dict]) -> str:
        """Provide guidance on self-healing capabilities"""
        guidance = [
            f"I'm the {self.name} agent. I can help with:",
            "• Analyzing system code health and detecting fragile patterns",
            "• Identifying architectural flaws and design smells",
            "• Generating specific repair goals for system improvement",
            "• Running comprehensive diagnostic suites",
            "• Prioritizing repairs by impact and feasibility",
            "",
            "Try: 'diagnose system' or 'generate repair goals' to get started"
        ]
        
        return "\n".join(guidance)