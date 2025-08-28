#!/usr/bin/env python3
"""
CodeRefactorer Agent for MeRNSTA Self-Upgrading System

Executes refactoring operations based on ArchitectAnalyzer suggestions.
Uses LLM reasoning to propose and implement code improvements.
"""

import os
import ast
import logging
import shutil
import hashlib
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .base import BaseAgent


class CodeRefactorer(BaseAgent):
    """
    Agent that executes code refactoring based on architectural analysis.
    
    Capabilities:
    - Proposes new designs for problematic code
    - Splits large files and classes
    - Introduces abstraction layers
    - Validates refactored code through testing
    - Manages safe deployment of changes
    """
    
    def __init__(self):
        super().__init__("code_refactorer")
        self.project_root = Path(os.getcwd())
        self.refactor_staging = self.project_root / "core_v2"
        self.backup_dir = self.project_root / "backups"
        
        # Ensure staging directories exist
        self.refactor_staging.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
    
    def get_agent_instructions(self) -> str:
        """Return specialized instructions for the code refactorer agent."""
        return (
            "You are a code refactoring specialist focused on implementing architectural improvements. "
            "Your role is to execute refactoring suggestions safely, including splitting god classes, "
            "resolving circular imports, extracting common patterns, and improving code structure. "
            "You create backups, validate syntax, run tests, and ensure refactored code maintains "
            "functionality while improving maintainability. Focus on safe transformation techniques "
            "and automated validation to deliver reliable refactoring results."
        )
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """Generate refactoring responses and execute refactoring tasks."""
        context = context or {}
        
        # Build memory context for refactoring patterns
        memory_context = self.get_memory_context(message)
        
        # Use LLM if available for complex refactoring guidance
        if self.llm_fallback:
            prompt = self.build_agent_prompt(message, memory_context)
            try:
                return self.llm_fallback.process(prompt)
            except Exception as e:
                logging.error(f"[{self.name}] LLM processing failed: {e}")
        
        # Handle refactoring execution requests
        if "refactor" in message.lower() or "execute" in message.lower():
            if "suggestion" in context:
                try:
                    result = self.execute_refactor(context["suggestion"])
                    if result.get("success"):
                        return f"Refactoring completed successfully! Applied {len(result.get('changes', []))} changes."
                    else:
                        return f"Refactoring failed: {'; '.join(result.get('errors', ['Unknown error']))}"
                except Exception as e:
                    return f"Refactoring execution failed: {str(e)}"
            else:
                return "Please provide a refactoring suggestion to execute."
        
        return "I can execute code refactoring based on architectural analysis suggestions. Provide a refactoring suggestion or ask about refactoring techniques."
    
    def execute_refactor(self, suggestion: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a refactoring suggestion from the ArchitectAnalyzer.
        
        Args:
            suggestion: Structured suggestion from ArchitectAnalyzer
            
        Returns:
            Refactoring result with success status and details
        """
        logging.info(f"[{self.name}] Executing refactor: {suggestion.get('title', 'Unknown')}")
        
        refactor_result = {
            "suggestion_id": suggestion.get("id"),
            "type": suggestion.get("type"),
            "started_at": datetime.now().isoformat(),
            "success": False,
            "changes": [],
            "errors": [],
            "test_results": None,
            "backup_location": None
        }
        
        try:
            # Create backup of affected modules
            backup_path = self._create_backup(suggestion.get("affected_modules", []))
            refactor_result["backup_location"] = str(backup_path)
            
            # Execute refactoring based on type
            refactor_type = suggestion.get("type")
            
            if refactor_type == "god_class":
                changes = self._refactor_god_class(suggestion)
            elif refactor_type == "god_module":
                changes = self._refactor_god_module(suggestion)
            elif refactor_type == "circular_import":
                changes = self._resolve_circular_imports(suggestion)
            elif refactor_type == "duplicate_pattern":
                changes = self._abstract_duplicate_pattern(suggestion)
            else:
                raise ValueError(f"Unknown refactor type: {refactor_type}")
            
            refactor_result["changes"] = changes
            
            # Validate syntax of all changed files
            syntax_valid = self._validate_syntax(changes)
            if not syntax_valid:
                refactor_result["errors"].append("Syntax validation failed")
                return refactor_result
            
            # Run tests if available
            test_results = self._run_tests(changes)
            refactor_result["test_results"] = test_results
            
            if test_results.get("passed", False):
                # Promote changes from staging to main codebase
                self._promote_changes(changes)
                refactor_result["success"] = True
                logging.info(f"[{self.name}] Refactor completed successfully")
            else:
                refactor_result["errors"].append("Tests failed after refactoring")
                
        except Exception as e:
            logging.error(f"[{self.name}] Refactor failed: {e}")
            refactor_result["errors"].append(str(e))
        
        refactor_result["completed_at"] = datetime.now().isoformat()
        return refactor_result
    
    def _create_backup(self, affected_modules: List[str]) -> Path:
        """Create backup of modules before refactoring."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"refactor_backup_{timestamp}"
        backup_path.mkdir(exist_ok=True)
        
        for module_path in affected_modules:
            source_file = self.project_root / module_path
            if source_file.exists():
                dest_file = backup_path / module_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, dest_file)
        
        logging.info(f"[{self.name}] Created backup at {backup_path}")
        return backup_path
    
    def _refactor_god_class(self, suggestion: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Refactor a god class by splitting it into smaller classes."""
        affected_modules = suggestion.get("affected_modules", [])
        if not affected_modules:
            raise ValueError("No affected modules specified")
            
        main_module = affected_modules[0]
        source_file = self.project_root / main_module
        
        with open(source_file, 'r') as f:
            source_code = f.read()
        
        # Parse the code to find the god class
        tree = ast.parse(source_code)
        god_class_name = suggestion.get("title", "").split()[-1]  # Extract class name from title
        
        # Use LLM to propose refactoring
        refactor_proposal = self._generate_class_refactor_proposal(source_code, god_class_name)
        
        # Create new files based on proposal
        changes = []
        for new_class in refactor_proposal.get("new_classes", []):
            new_file_path = self.refactor_staging / new_class["filename"]
            new_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(new_file_path, 'w') as f:
                f.write(new_class["code"])
            
            changes.append({
                "type": "create",
                "path": str(new_file_path),
                "content": new_class["code"]
            })
        
        # Update the original file
        updated_main_code = refactor_proposal.get("updated_main_file", source_code)
        updated_file_path = self.refactor_staging / main_module
        updated_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(updated_file_path, 'w') as f:
            f.write(updated_main_code)
        
        changes.append({
            "type": "modify",
            "path": str(updated_file_path),
            "original_path": str(source_file),
            "content": updated_main_code
        })
        
        return changes
    
    def _refactor_god_module(self, suggestion: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Refactor a god module by splitting it into focused modules."""
        affected_modules = suggestion.get("affected_modules", [])
        if not affected_modules:
            raise ValueError("No affected modules specified")
            
        main_module = affected_modules[0]
        source_file = self.project_root / main_module
        
        with open(source_file, 'r') as f:
            source_code = f.read()
        
        # Use LLM to propose module splitting
        refactor_proposal = self._generate_module_split_proposal(source_code, main_module)
        
        changes = []
        
        # Create new modules based on proposal
        for new_module in refactor_proposal.get("new_modules", []):
            new_file_path = self.refactor_staging / new_module["filename"]
            new_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(new_file_path, 'w') as f:
                f.write(new_module["code"])
            
            changes.append({
                "type": "create",
                "path": str(new_file_path),
                "content": new_module["code"]
            })
        
        # Update the main module to use new modules
        updated_main_code = refactor_proposal.get("updated_main_module", source_code)
        updated_file_path = self.refactor_staging / main_module
        updated_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(updated_file_path, 'w') as f:
            f.write(updated_main_code)
        
        changes.append({
            "type": "modify",
            "path": str(updated_file_path),
            "original_path": str(source_file),
            "content": updated_main_code
        })
        
        return changes
    
    def _resolve_circular_imports(self, suggestion: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Resolve circular import dependencies."""
        cycle = suggestion.get("affected_modules", [])
        if len(cycle) < 2:
            raise ValueError("Invalid circular import cycle")
        
        changes = []
        
        # Analyze the circular dependency
        dependency_analysis = self._analyze_circular_dependency(cycle)
        
        # Use LLM to propose resolution strategy
        resolution_proposal = self._generate_circular_import_resolution(dependency_analysis)
        
        # Apply the resolution strategy
        for module_path, new_code in resolution_proposal.get("updated_modules", {}).items():
            updated_file_path = self.refactor_staging / module_path
            updated_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(updated_file_path, 'w') as f:
                f.write(new_code)
            
            changes.append({
                "type": "modify",
                "path": str(updated_file_path),
                "original_path": str(self.project_root / module_path),
                "content": new_code
            })
        
        # Create new interface modules if needed
        for interface_module in resolution_proposal.get("new_interfaces", []):
            new_file_path = self.refactor_staging / interface_module["filename"]
            new_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(new_file_path, 'w') as f:
                f.write(interface_module["code"])
            
            changes.append({
                "type": "create",
                "path": str(new_file_path),
                "content": interface_module["code"]
            })
        
        return changes
    
    def _abstract_duplicate_pattern(self, suggestion: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create abstraction for duplicate code patterns."""
        instances = suggestion.get("affected_modules", [])
        if len(instances) < 2:
            raise ValueError("Need at least 2 instances for abstraction")
        
        # Collect code from all instances
        instance_codes = []
        for module_path in instances:
            source_file = self.project_root / module_path
            if source_file.exists():
                with open(source_file, 'r') as f:
                    instance_codes.append({
                        "module": module_path,
                        "code": f.read()
                    })
        
        # Use LLM to create abstraction
        abstraction_proposal = self._generate_abstraction_proposal(instance_codes)
        
        changes = []
        
        # Create the new abstraction module
        abstraction_module = abstraction_proposal.get("abstraction_module")
        if abstraction_module:
            new_file_path = self.refactor_staging / abstraction_module["filename"]
            new_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(new_file_path, 'w') as f:
                f.write(abstraction_module["code"])
            
            changes.append({
                "type": "create",
                "path": str(new_file_path),
                "content": abstraction_module["code"]
            })
        
        # Update all instances to use the abstraction
        for updated_instance in abstraction_proposal.get("updated_instances", []):
            updated_file_path = self.refactor_staging / updated_instance["module"]
            updated_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(updated_file_path, 'w') as f:
                f.write(updated_instance["code"])
            
            changes.append({
                "type": "modify",
                "path": str(updated_file_path),
                "original_path": str(self.project_root / updated_instance["module"]),
                "content": updated_instance["code"]
            })
        
        return changes
    
    def _generate_class_refactor_proposal(self, source_code: str, class_name: str) -> Dict[str, Any]:
        """Use LLM to generate class refactoring proposal."""
        if not self.llm_fallback:
            # Fallback to simple splitting strategy
            return self._simple_class_split(source_code, class_name)
        
        prompt = f"""
        Analyze this Python code and propose a refactoring for the class '{class_name}' which appears to be a god class.
        
        Source code:
        ```python
        {source_code}
        ```
        
        Please provide a refactoring proposal that:
        1. Splits the god class into smaller, focused classes
        2. Maintains existing functionality
        3. Follows single responsibility principle
        4. Provides clear interfaces between classes
        
        Return your proposal as JSON with this structure:
        {{
            "new_classes": [
                {{
                    "filename": "path/to/new_file.py",
                    "class_name": "NewClassName",
                    "code": "complete Python code for the new file"
                }}
            ],
            "updated_main_file": "updated code for the original file",
            "rationale": "explanation of the refactoring approach"
        }}
        """
        
        try:
            response = self.llm_fallback.process(prompt)
            # Parse JSON response (in a real implementation, you'd handle this more robustly)
            import json
            proposal = json.loads(response)
            return proposal
        except Exception as e:
            logging.error(f"[{self.name}] LLM refactor proposal failed: {e}")
            return self._simple_class_split(source_code, class_name)
    
    def _generate_module_split_proposal(self, source_code: str, module_path: str) -> Dict[str, Any]:
        """Use LLM to generate module splitting proposal."""
        if not self.llm_fallback:
            return self._simple_module_split(source_code, module_path)
        
        prompt = f"""
        Analyze this Python module that appears to be too large and propose a way to split it into smaller, focused modules.
        
        Module: {module_path}
        Source code:
        ```python
        {source_code}
        ```
        
        Please provide a refactoring proposal that:
        1. Groups related functions and classes together
        2. Minimizes dependencies between new modules
        3. Maintains all existing functionality
        4. Creates clear module boundaries
        
        Return your proposal as JSON with this structure:
        {{
            "new_modules": [
                {{
                    "filename": "path/to/new_module.py",
                    "purpose": "description of module purpose",
                    "code": "complete Python code for the new module"
                }}
            ],
            "updated_main_module": "updated code for the original module",
            "rationale": "explanation of the splitting approach"
        }}
        """
        
        try:
            response = self.llm_fallback.process(prompt)
            import json
            proposal = json.loads(response)
            return proposal
        except Exception as e:
            logging.error(f"[{self.name}] LLM module split proposal failed: {e}")
            return self._simple_module_split(source_code, module_path)
    
    def _generate_circular_import_resolution(self, dependency_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to generate circular import resolution."""
        if not self.llm_fallback:
            return self._simple_circular_resolution(dependency_analysis)
        
        prompt = f"""
        Analyze this circular import dependency and propose a resolution.
        
        Dependency analysis:
        {json.dumps(dependency_analysis, indent=2)}
        
        Please provide a resolution that:
        1. Breaks the circular dependency
        2. Maintains all existing functionality
        3. Uses good architectural patterns (dependency injection, interfaces, etc.)
        4. Minimizes code changes
        
        Return your proposal as JSON with this structure:
        {{
            "strategy": "description of resolution strategy",
            "updated_modules": {{
                "module_path": "updated code for module"
            }},
            "new_interfaces": [
                {{
                    "filename": "path/to/interface.py", 
                    "code": "interface/abstract base class code"
                }}
            ],
            "rationale": "explanation of the approach"
        }}
        """
        
        try:
            response = self.llm_fallback.process(prompt)
            import json
            proposal = json.loads(response)
            return proposal
        except Exception as e:
            logging.error(f"[{self.name}] LLM circular import resolution failed: {e}")
            return self._simple_circular_resolution(dependency_analysis)
    
    def _generate_abstraction_proposal(self, instance_codes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use LLM to generate abstraction for duplicate patterns."""
        if not self.llm_fallback:
            return self._simple_abstraction(instance_codes)
        
        codes_text = "\n\n".join([f"Module: {inst['module']}\n```python\n{inst['code']}\n```" 
                                  for inst in instance_codes])
        
        prompt = f"""
        Analyze these code instances that contain duplicate patterns and create a shared abstraction.
        
        Code instances:
        {codes_text}
        
        Please provide an abstraction that:
        1. Captures the common pattern
        2. Allows for variation in the instances
        3. Reduces code duplication
        4. Maintains existing functionality
        
        Return your proposal as JSON with this structure:
        {{
            "abstraction_module": {{
                "filename": "path/to/abstraction.py",
                "code": "code for the shared abstraction"
            }},
            "updated_instances": [
                {{
                    "module": "module_path",
                    "code": "updated code using the abstraction"
                }}
            ],
            "rationale": "explanation of the abstraction approach"
        }}
        """
        
        try:
            response = self.llm_fallback.process(prompt)
            import json
            proposal = json.loads(response)
            return proposal
        except Exception as e:
            logging.error(f"[{self.name}] LLM abstraction proposal failed: {e}")
            return self._simple_abstraction(instance_codes)
    
    def _simple_class_split(self, source_code: str, class_name: str) -> Dict[str, Any]:
        """Simple fallback for class splitting without LLM."""
        # Basic strategy: move methods to a helper class
        return {
            "new_classes": [{
                "filename": f"utils/{class_name.lower()}_helper.py",
                "class_name": f"{class_name}Helper",
                "code": f"# Helper class for {class_name}\nclass {class_name}Helper:\n    pass\n"
            }],
            "updated_main_file": source_code,  # No changes for now
            "rationale": "Simple helper class extraction (fallback strategy)"
        }
    
    def _simple_module_split(self, source_code: str, module_path: str) -> Dict[str, Any]:
        """Simple fallback for module splitting without LLM."""
        base_name = Path(module_path).stem
        return {
            "new_modules": [{
                "filename": f"{Path(module_path).parent}/{base_name}_utils.py",
                "purpose": "Utility functions",
                "code": f"# Utilities for {base_name}\n"
            }],
            "updated_main_module": source_code,  # No changes for now
            "rationale": "Simple utility module extraction (fallback strategy)"
        }
    
    def _simple_circular_resolution(self, dependency_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Simple fallback for circular import resolution."""
        return {
            "strategy": "Move shared definitions to common module",
            "updated_modules": {},
            "new_interfaces": [],
            "rationale": "Fallback strategy - manual resolution needed"
        }
    
    def _simple_abstraction(self, instance_codes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple fallback for creating abstractions."""
        return {
            "abstraction_module": {
                "filename": "utils/common_patterns.py",
                "code": "# Common patterns abstraction\n"
            },
            "updated_instances": [],
            "rationale": "Fallback strategy - manual abstraction needed"
        }
    
    def _analyze_circular_dependency(self, cycle: List[str]) -> Dict[str, Any]:
        """Analyze what causes the circular dependency."""
        analysis = {
            "cycle": cycle,
            "dependencies": {},
            "shared_symbols": []
        }
        
        # For each module in the cycle, find what it imports from the next module
        for i, module_path in enumerate(cycle[:-1]):  # Exclude last duplicate
            current_module = self.project_root / module_path
            next_module = cycle[i + 1]
            
            if current_module.exists():
                with open(current_module, 'r') as f:
                    code = f.read()
                
                # Simple analysis of imports (could be made more sophisticated)
                imports = []
                for line in code.split('\n'):
                    if f'from {next_module.replace("/", ".").replace(".py", "")}' in line:
                        imports.append(line.strip())
                    elif f'import {next_module.replace("/", ".").replace(".py", "")}' in line:
                        imports.append(line.strip())
                
                analysis["dependencies"][module_path] = imports
        
        return analysis
    
    def _validate_syntax(self, changes: List[Dict[str, Any]]) -> bool:
        """Validate syntax of all changed Python files."""
        for change in changes:
            if change["path"].endswith(".py"):
                try:
                    ast.parse(change["content"])
                except SyntaxError as e:
                    logging.error(f"[{self.name}] Syntax error in {change['path']}: {e}")
                    return False
        return True
    
    def _run_tests(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run tests to validate the refactored code."""
        test_result = {
            "passed": False,
            "output": "",
            "affected_tests": []
        }
        
        try:
            # Run pytest on the staging directory
            cmd = ["python", "-m", "pytest", str(self.refactor_staging), "-v"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            test_result["output"] = result.stdout + result.stderr
            test_result["passed"] = result.returncode == 0
            
            # Also run syntax check
            if test_result["passed"]:
                for change in changes:
                    if change["path"].endswith(".py"):
                        cmd = ["python", "-m", "py_compile", change["path"]]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.returncode != 0:
                            test_result["passed"] = False
                            test_result["output"] += f"\nSyntax error in {change['path']}: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            test_result["output"] = "Tests timed out after 5 minutes"
        except Exception as e:
            test_result["output"] = f"Test execution failed: {e}"
        
        return test_result
    
    def _promote_changes(self, changes: List[Dict[str, Any]]) -> None:
        """Promote validated changes from staging to main codebase."""
        for change in changes:
            if change["type"] == "create":
                # Copy new file to main codebase
                staging_path = Path(change["path"])
                relative_path = staging_path.relative_to(self.refactor_staging)
                target_path = self.project_root / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(staging_path, target_path)
                
            elif change["type"] == "modify":
                # Replace original file with modified version
                staging_path = Path(change["path"])
                original_path = Path(change["original_path"])
                shutil.copy2(staging_path, original_path)
        
        logging.info(f"[{self.name}] Promoted {len(changes)} changes to main codebase")