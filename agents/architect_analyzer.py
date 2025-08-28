#!/usr/bin/env python3
"""
ArchitectAnalyzer Agent for MeRNSTA Self-Upgrading System

Analyzes codebase for architectural flaws and suggests improvements.
Scans for monoliths, coupling issues, circular imports, and refactoring opportunities.
"""

import ast
import os
import sys
import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict, Counter

from .base import BaseAgent


class ArchitectAnalyzer(BaseAgent):
    """
    Agent that analyzes codebase architecture and identifies improvement opportunities.
    
    Capabilities:
    - Detects monoliths and god classes
    - Identifies circular imports
    - Measures coupling and cohesion
    - Finds repeated patterns for abstraction
    - Analyzes control flow inefficiencies
    """
    
    def __init__(self):
        super().__init__("architect_analyzer")
        self.analysis_cache = {}
        self.project_root = Path(os.getcwd())
    
    def get_agent_instructions(self) -> str:
        """Return specialized instructions for the architect analyzer agent."""
        return (
            "You are an architectural analysis specialist focused on codebase quality and design patterns. "
            "Your role is to analyze code architecture, detect architectural flaws like god classes, "
            "circular imports, and coupling issues. You identify refactoring opportunities, measure "
            "code complexity, and suggest architectural improvements. Focus on modularity, maintainability, "
            "and design principles to provide actionable recommendations for code structure improvements."
        )
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """Generate architectural analysis responses."""
        context = context or {}
        
        # Build memory context for architectural patterns
        memory_context = self.get_memory_context(message)
        
        # Use LLM if available for complex analysis questions
        if self.llm_fallback:
            prompt = self.build_agent_prompt(message, memory_context)
            try:
                return self.llm_fallback.process(prompt)
            except Exception as e:
                logging.error(f"[{self.name}] LLM processing failed: {e}")
        
        # Fallback to programmatic analysis
        if "analyze" in message.lower() and "codebase" in message.lower():
            try:
                analysis_results = self.analyze_codebase()
                suggestions_count = len(analysis_results.get("upgrade_suggestions", []))
                return f"Completed architectural analysis. Found {suggestions_count} improvement suggestions. Check analysis results for detailed recommendations."
            except Exception as e:
                return f"Analysis failed: {str(e)}"
        
        return "I can analyze your codebase architecture. Try asking me to 'analyze codebase' or ask specific questions about architectural patterns and code structure."
        
    def analyze_codebase(self, target_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive architectural analysis of the codebase.
        
        Args:
            target_path: Specific path to analyze, defaults to entire project
            
        Returns:
            Structured analysis report with upgrade suggestions
        """
        logging.info(f"[{self.name}] Starting codebase analysis...")
        
        if target_path:
            analysis_root = Path(target_path)
        else:
            analysis_root = self.project_root
            
        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "analyzed_path": str(analysis_root),
            "modules": {},
            "global_issues": {},
            "upgrade_suggestions": []
        }
        
        # Collect all Python files
        python_files = self._collect_python_files(analysis_root)
        logging.info(f"[{self.name}] Found {len(python_files)} Python files to analyze")
        
        # Analyze each file
        for file_path in python_files:
            module_analysis = self._analyze_module(file_path)
            relative_path = str(file_path.relative_to(self.project_root))
            analysis_results["modules"][relative_path] = module_analysis
            
        # Perform global analysis
        analysis_results["global_issues"] = self._analyze_global_patterns(analysis_results["modules"])
        
        # Generate upgrade suggestions
        analysis_results["upgrade_suggestions"] = self._generate_upgrade_suggestions(analysis_results)
        
        logging.info(f"[{self.name}] Analysis complete. Found {len(analysis_results['upgrade_suggestions'])} suggestions")
        return analysis_results
    
    def _collect_python_files(self, root_path: Path) -> List[Path]:
        """Collect all Python files in the given path."""
        python_files = []
        exclude_dirs = {'.git', '__pycache__', '.pytest_cache', '.venv', 'venv', 'node_modules'}
        
        for file_path in root_path.rglob("*.py"):
            # Skip files in excluded directories
            if any(exc_dir in file_path.parts for exc_dir in exclude_dirs):
                continue
            python_files.append(file_path)
            
        return python_files
    
    def _analyze_module(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python module for architectural issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
                
            # Parse AST
            try:
                tree = ast.parse(source_code)
            except SyntaxError as e:
                return {
                    "error": f"Syntax error: {e}",
                    "analyzable": False
                }
                
            analyzer = ModuleAnalyzer(file_path, source_code, tree)
            return analyzer.analyze()
            
        except Exception as e:
            logging.error(f"[{self.name}] Error analyzing {file_path}: {e}")
            return {
                "error": str(e),
                "analyzable": False
            }
    
    def _analyze_global_patterns(self, modules: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns across all modules."""
        global_issues = {
            "circular_imports": self._detect_circular_imports(modules),
            "duplicate_patterns": self._detect_duplicate_patterns(modules),
            "cross_module_coupling": self._analyze_cross_module_coupling(modules),
            "architectural_violations": self._detect_architectural_violations(modules)
        }
        
        return global_issues
    
    def _detect_circular_imports(self, modules: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect circular import dependencies."""
        import_graph = {}
        
        for module_path, module_data in modules.items():
            if not module_data.get("analyzable", True):
                continue
                
            imports = module_data.get("imports", [])
            import_graph[module_path] = []
            
            for imp in imports:
                # Convert import to module path
                module_name = imp.get("module", "")
                if module_name.startswith("."):
                    # Relative import
                    base_path = Path(module_path).parent
                    target_path = str(base_path / f"{module_name[1:]}.py")
                    import_graph[module_path].append(target_path)
                elif module_name in ["agents.", "storage.", "cortex.", "tools."]:
                    # Internal module
                    target_path = f"{module_name.replace('.', '/')}.py"
                    import_graph[module_path].append(target_path)
        
        # Find cycles using DFS
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append({
                    "cycle": cycle,
                    "severity": "high" if len(cycle) <= 3 else "medium"
                })
                return
                
            if node in visited or node not in import_graph:
                return
                
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in import_graph[node]:
                dfs(neighbor, path.copy())
                
            rec_stack.remove(node)
        
        for module in import_graph:
            if module not in visited:
                dfs(module, [])
        
        return cycles
    
    def _detect_duplicate_patterns(self, modules: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect duplicate code patterns that could be abstracted."""
        pattern_groups = defaultdict(list)
        
        for module_path, module_data in modules.items():
            if not module_data.get("analyzable", True):
                continue
                
            functions = module_data.get("functions", [])
            classes = module_data.get("classes", [])
            
            # Group similar functions by signature pattern
            for func in functions:
                signature_pattern = self._normalize_function_signature(func)
                pattern_groups[f"func_{signature_pattern}"].append({
                    "module": module_path,
                    "name": func["name"],
                    "type": "function",
                    "lines": func.get("lines", 0)
                })
                
            # Group similar classes by method patterns
            for cls in classes:
                method_pattern = self._normalize_class_methods(cls)
                pattern_groups[f"class_{method_pattern}"].append({
                    "module": module_path,
                    "name": cls["name"],
                    "type": "class",
                    "lines": cls.get("lines", 0)
                })
        
        # Find patterns that appear multiple times
        duplicates = []
        for pattern, instances in pattern_groups.items():
            if len(instances) >= 2:
                duplicates.append({
                    "pattern": pattern,
                    "instances": instances,
                    "potential_savings": sum(inst.get("lines", 0) for inst in instances[1:])
                })
        
        return sorted(duplicates, key=lambda x: x["potential_savings"], reverse=True)
    
    def _analyze_cross_module_coupling(self, modules: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coupling between modules."""
        coupling_matrix = defaultdict(int)
        
        for module_path, module_data in modules.items():
            if not module_data.get("analyzable", True):
                continue
                
            imports = module_data.get("imports", [])
            for imp in imports:
                module_name = imp.get("module", "")
                if any(module_name.startswith(prefix) for prefix in ["agents.", "storage.", "cortex.", "tools."]):
                    coupling_matrix[f"{module_path} -> {module_name}"] += 1
        
        # Identify high-coupling modules
        high_coupling = []
        coupling_counts = defaultdict(int)
        
        for edge, weight in coupling_matrix.items():
            source = edge.split(" -> ")[0]
            coupling_counts[source] += weight
            
        for module, count in coupling_counts.items():
            if count > 10:  # Threshold for high coupling
                high_coupling.append({
                    "module": module,
                    "coupling_count": count,
                    "severity": "high" if count > 20 else "medium"
                })
        
        return {
            "coupling_matrix": dict(coupling_matrix),
            "high_coupling_modules": sorted(high_coupling, key=lambda x: x["coupling_count"], reverse=True)
        }
    
    def _detect_architectural_violations(self, modules: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect violations of architectural principles."""
        violations = []
        
        for module_path, module_data in modules.items():
            if not module_data.get("analyzable", True):
                continue
                
            # Check for god classes (classes with too many methods/lines)
            classes = module_data.get("classes", [])
            for cls in classes:
                if cls.get("methods", 0) > 20 or cls.get("lines", 0) > 500:
                    violations.append({
                        "type": "god_class",
                        "module": module_path,
                        "class": cls["name"],
                        "methods": cls.get("methods", 0),
                        "lines": cls.get("lines", 0),
                        "severity": "high" if cls.get("lines", 0) > 1000 else "medium"
                    })
            
            # Check for god modules (modules with too many functions/lines)
            total_functions = len(module_data.get("functions", []))
            total_lines = module_data.get("total_lines", 0)
            
            if total_functions > 30 or total_lines > 1000:
                violations.append({
                    "type": "god_module",
                    "module": module_path,
                    "functions": total_functions,
                    "lines": total_lines,
                    "severity": "high" if total_lines > 2000 else "medium"
                })
        
        return violations
    
    def _generate_upgrade_suggestions(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate prioritized upgrade suggestions based on analysis."""
        suggestions = []
        
        # Suggestions from global issues
        global_issues = analysis_results.get("global_issues", {})
        
        # Circular import suggestions
        for cycle in global_issues.get("circular_imports", []):
            suggestions.append({
                "id": f"circular_import_{hashlib.md5(str(cycle['cycle']).encode()).hexdigest()[:8]}",
                "type": "circular_import",
                "title": f"Resolve circular import cycle",
                "description": f"Break circular dependency: {' -> '.join(cycle['cycle'])}",
                "affected_modules": cycle["cycle"],
                "risk_level": cycle["severity"],
                "refactor_difficulty": "medium",
                "proposed_benefit": "Improved modularity and testability",
                "priority": 8 if cycle["severity"] == "high" else 6
            })
        
        # God class/module suggestions
        for violation in global_issues.get("architectural_violations", []):
            if violation["type"] == "god_class":
                suggestions.append({
                    "id": f"god_class_{hashlib.md5(f'{violation['module']}_{violation['class']}'.encode()).hexdigest()[:8]}",
                    "type": "god_class",
                    "title": f"Refactor god class {violation['class']}",
                    "description": f"Split large class with {violation['methods']} methods into smaller, focused classes",
                    "affected_modules": [violation["module"]],
                    "risk_level": violation["severity"],
                    "refactor_difficulty": "high",
                    "proposed_benefit": "Better maintainability and single responsibility",
                    "priority": 7 if violation["severity"] == "high" else 5
                })
            elif violation["type"] == "god_module":
                suggestions.append({
                    "id": f"god_module_{hashlib.md5(violation['module'].encode()).hexdigest()[:8]}",
                    "type": "god_module", 
                    "title": f"Split large module {violation['module']}",
                    "description": f"Break down module with {violation['functions']} functions into focused modules",
                    "affected_modules": [violation["module"]],
                    "risk_level": violation["severity"],
                    "refactor_difficulty": "medium",
                    "proposed_benefit": "Improved organization and readability",
                    "priority": 6 if violation["severity"] == "high" else 4
                })
        
        # Duplicate pattern suggestions
        for duplicate in global_issues.get("duplicate_patterns", [])[:5]:  # Top 5
            suggestions.append({
                "id": f"duplicate_{hashlib.md5(duplicate['pattern'].encode()).hexdigest()[:8]}",
                "type": "duplicate_pattern",
                "title": f"Abstract duplicate pattern",
                "description": f"Create shared abstraction for pattern found in {len(duplicate['instances'])} places",
                "affected_modules": [inst["module"] for inst in duplicate["instances"]],
                "risk_level": "low",
                "refactor_difficulty": "medium",
                "proposed_benefit": f"Reduce {duplicate['potential_savings']} lines of duplicate code",
                "priority": min(9, 3 + duplicate['potential_savings'] // 100)
            })
        
        # Sort by priority
        suggestions.sort(key=lambda x: x["priority"], reverse=True)
        
        return suggestions
    
    def _normalize_function_signature(self, func: Dict[str, Any]) -> str:
        """Create a normalized signature pattern for function comparison."""
        args = func.get("args", [])
        return f"args_{len(args)}_returns_{func.get('returns_type', 'unknown')}"
    
    def _normalize_class_methods(self, cls: Dict[str, Any]) -> str:
        """Create a normalized method pattern for class comparison."""
        method_count = cls.get("methods", 0)
        return f"methods_{method_count}_props_{cls.get('properties', 0)}"


class ModuleAnalyzer(ast.NodeVisitor):
    """AST visitor for analyzing individual Python modules."""
    
    def __init__(self, file_path: Path, source_code: str, tree: ast.AST):
        self.file_path = file_path
        self.source_code = source_code
        self.tree = tree
        self.lines = source_code.split('\n')
        
        # Analysis results
        self.imports = []
        self.functions = []
        self.classes = []
        self.complexity_score = 0
        
    def analyze(self) -> Dict[str, Any]:
        """Perform the analysis and return results."""
        self.visit(self.tree)
        
        return {
            "analyzable": True,
            "total_lines": len(self.lines),
            "imports": self.imports,
            "functions": self.functions,
            "classes": self.classes,
            "complexity_score": self.complexity_score,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def visit_Import(self, node: ast.Import):
        """Visit import statements."""
        for alias in node.names:
            self.imports.append({
                "type": "import",
                "module": alias.name,
                "alias": alias.asname,
                "line": node.lineno
            })
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Visit from ... import statements."""
        module_name = node.module or ""
        for alias in node.names:
            self.imports.append({
                "type": "from_import",
                "module": module_name,
                "name": alias.name,
                "alias": alias.asname,
                "line": node.lineno
            })
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definitions."""
        # Calculate function metrics
        func_lines = (node.end_lineno or node.lineno) - node.lineno + 1
        
        func_info = {
            "name": node.name,
            "line": node.lineno,
            "lines": func_lines,
            "args": [arg.arg for arg in node.args.args],
            "returns_type": self._extract_return_type(node),
            "complexity": self._calculate_function_complexity(node)
        }
        
        self.functions.append(func_info)
        self.complexity_score += func_info["complexity"]
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definitions."""
        self.visit_FunctionDef(node)  # Same analysis as regular functions
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definitions."""
        # Count methods and properties
        methods = 0
        properties = 0
        
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods += 1
            elif isinstance(item, ast.AnnAssign):
                properties += 1
        
        class_lines = (node.end_lineno or node.lineno) - node.lineno + 1
        
        class_info = {
            "name": node.name,
            "line": node.lineno,
            "lines": class_lines,
            "methods": methods,
            "properties": properties,
            "bases": [self._extract_base_name(base) for base in node.bases]
        }
        
        self.classes.append(class_info)
        self.generic_visit(node)
    
    def _extract_return_type(self, node: ast.FunctionDef) -> str:
        """Extract return type annotation if present."""
        if node.returns:
            return ast.unparse(node.returns) if hasattr(ast, 'unparse') else "annotated"
        return "unknown"
    
    def _extract_base_name(self, base: ast.expr) -> str:
        """Extract base class name."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return f"{self._extract_base_name(base.value)}.{base.attr}"
        return "unknown"
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.Try):
                complexity += len(child.handlers)
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity