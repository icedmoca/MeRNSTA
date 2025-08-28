#!/usr/bin/env python3
"""
Mutation Utilities for Agent Self-Replication

Handles code-level mutations including:
- Function/class renaming
- Logic tweaks and prompt adjustments
- Syntax validation
- Code pattern modifications
"""

import ast
import re
import random
import logging
import shutil
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


class MutationEngine:
    """
    Engine for applying code mutations to agent files.
    
    Supports various mutation types while maintaining syntactic validity.
    """
    
    def __init__(self, mutation_rate: float = 0.2):
        self.mutation_rate = mutation_rate
        
        # Mutation strategies
        self.mutation_strategies = [
            self._mutate_function_names,
            self._mutate_class_names,
            self._mutate_string_literals,
            self._mutate_numeric_constants,
            self._mutate_prompt_strings,
            self._mutate_logic_conditions,
            self._mutate_variable_names,
            self._mutate_control_flow,
            self._mutate_error_messages,
            self._mutate_timeout_values
        ]
        
        # Safe function name variations
        self.function_name_variations = {
            'respond': ['handle', 'process', 'analyze', 'evaluate'],
            'get_agent_instructions': ['get_instructions', 'get_guidance', 'get_directions'],
            'analyze': ['evaluate', 'assess', 'examine', 'process'],
            'process': ['handle', 'manage', 'execute', 'run'],
            'evaluate': ['assess', 'analyze', 'score', 'rate'],
            'generate': ['create', 'produce', 'build', 'make'],
            'handle': ['process', 'manage', 'deal_with', 'execute']
        }
        
        # Safe class name variations
        self.class_name_variations = {
            'Agent': ['Handler', 'Processor', 'Analyzer', 'Manager'],
            'Critic': ['Evaluator', 'Assessor', 'Reviewer', 'Analyzer'],
            'Planner': ['Strategist', 'Organizer', 'Coordinator', 'Designer'],
            'Debater': ['Arguer', 'Discusser', 'Challenger', 'Questioner']
        }
        
        # Prompt mutation templates
        self.prompt_mutations = [
            lambda s: s.replace('analyze', 'evaluate'),
            lambda s: s.replace('carefully', 'thoroughly'),
            lambda s: s.replace('consider', 'examine'),
            lambda s: s.replace('provide', 'give'),
            lambda s: s.replace('detailed', 'comprehensive'),
            lambda s: s.replace('explain', 'describe'),
            lambda s: s.replace('focus on', 'emphasize'),
            lambda s: s.replace('important', 'crucial'),
            lambda s: s.replace('ensure', 'make sure'),
            lambda s: s.replace('identify', 'find')
        ]
        
        logging.info(f"[MutationEngine] Initialized with mutation_rate={mutation_rate}")
    
    def mutate_file(self, file_path: str) -> bool:
        """
        Apply mutations to a Python file
        
        Args:
            file_path: Path to the file to mutate
            
        Returns:
            True if mutations applied successfully, False otherwise
        """
        
        try:
            file_path = Path(file_path)
            
            # Read original content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Parse AST to ensure it's valid Python
            try:
                ast.parse(original_content)
            except SyntaxError as e:
                logging.error(f"[MutationEngine] Invalid syntax in {file_path}: {e}")
                return False
            
            # Apply mutations
            mutated_content = original_content
            mutations_applied = 0
            
            for strategy in self.mutation_strategies:
                if random.random() < self.mutation_rate:
                    try:
                        new_content = strategy(mutated_content)
                        
                        # Validate that mutation doesn't break syntax
                        ast.parse(new_content)
                        
                        mutated_content = new_content
                        mutations_applied += 1
                        
                    except (SyntaxError, Exception) as e:
                        logging.warning(f"[MutationEngine] Mutation failed, skipping: {e}")
                        continue
            
            # Only write if mutations were applied
            if mutations_applied > 0:
                # Final syntax check
                try:
                    ast.parse(mutated_content)
                except SyntaxError:
                    logging.error(f"[MutationEngine] Final syntax check failed for {file_path}")
                    return False
                
                # Write mutated content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(mutated_content)
                
                logging.info(f"[MutationEngine] Applied {mutations_applied} mutations to {file_path}")
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"[MutationEngine] Failed to mutate {file_path}: {e}")
            return False
    
    def _mutate_function_names(self, content: str) -> str:
        """Rename function definitions with variations"""
        
        lines = content.split('\n')
        mutated_lines = []
        
        for line in lines:
            # Match function definitions
            match = re.match(r'^(\s*)def\s+(\w+)(\(.*\):.*)', line)
            if match:
                indent, func_name, rest = match.groups()
                
                # Get variations for this function name
                variations = []
                for base_name, vars_list in self.function_name_variations.items():
                    if base_name in func_name.lower():
                        variations.extend(vars_list)
                
                if variations and random.random() < 0.3:  # 30% chance to rename
                    new_name = random.choice(variations)
                    # Preserve naming convention
                    if func_name.startswith('_'):
                        new_name = '_' + new_name
                    if func_name.endswith('_'):
                        new_name = new_name + '_'
                    
                    line = f"{indent}def {new_name}{rest}"
            
            mutated_lines.append(line)
        
        return '\n'.join(mutated_lines)
    
    def _mutate_class_names(self, content: str) -> str:
        """Rename class definitions with variations"""
        
        lines = content.split('\n')
        mutated_lines = []
        
        for line in lines:
            # Match class definitions
            match = re.match(r'^(\s*)class\s+(\w+)(\(.*\):.*|\:.*)', line)
            if match:
                indent, class_name, rest = match.groups()
                
                # Get variations for this class name
                variations = []
                for base_name, vars_list in self.class_name_variations.items():
                    if base_name in class_name:
                        variations.extend(vars_list)
                
                if variations and random.random() < 0.2:  # 20% chance to rename
                    new_name = random.choice(variations)
                    # Preserve naming convention (replace only the base part)
                    for base_name in self.class_name_variations.keys():
                        if base_name in class_name:
                            new_class_name = class_name.replace(base_name, new_name)
                            line = f"{indent}class {new_class_name}{rest}"
                            break
            
            mutated_lines.append(line)
        
        return '\n'.join(mutated_lines)
    
    def _mutate_string_literals(self, content: str) -> str:
        """Mutate string literals, especially docstrings and messages"""
        
        lines = content.split('\n')
        mutated_lines = []
        
        for line in lines:
            # Skip code lines that are not strings
            if not ('"' in line or "'" in line):
                mutated_lines.append(line)
                continue
            
            # Apply prompt mutations to string content
            if random.random() < 0.1:  # 10% chance to mutate strings
                for mutation in self.prompt_mutations:
                    if random.random() < 0.5:
                        try:
                            line = mutation(line)
                        except:
                            continue
            
            mutated_lines.append(line)
        
        return '\n'.join(mutated_lines)
    
    def _mutate_numeric_constants(self, content: str) -> str:
        """Slightly modify numeric constants"""
        
        def mutate_number(match):
            num_str = match.group()
            try:
                if '.' in num_str:
                    # Float
                    num = float(num_str)
                    # Apply small random change (±5%)
                    variation = random.uniform(-0.05, 0.05)
                    new_num = num * (1 + variation)
                    return f"{new_num:.3f}"
                else:
                    # Integer
                    num = int(num_str)
                    if num > 1:  # Don't mutate 0 or 1
                        variation = random.randint(-1, 1)
                        new_num = max(1, num + variation)
                        return str(new_num)
            except:
                pass
            return num_str
        
        # Match numeric constants (but not in imports or __version__ etc.)
        if random.random() < 0.1:  # 10% chance
            content = re.sub(r'(?<!\w)(\d+\.\d+|\d+)(?!\w)', mutate_number, content)
        
        return content
    
    def _mutate_prompt_strings(self, content: str) -> str:
        """Specifically target prompt strings and instructions"""
        
        lines = content.split('\n')
        mutated_lines = []
        
        in_docstring = False
        in_prompt = False
        
        for line in lines:
            # Detect docstrings and prompts
            if '"""' in line:
                in_docstring = not in_docstring
            
            # Look for prompt-like content
            if any(keyword in line.lower() for keyword in 
                   ['instruction', 'prompt', 'system', 'you are', 'your role']):
                in_prompt = True
            
            # Mutate prompt content
            if (in_docstring or in_prompt) and random.random() < 0.2:
                # Apply semantic mutations to prompts
                original_line = line
                for mutation in self.prompt_mutations:
                    if random.random() < 0.3:
                        try:
                            line = mutation(line)
                        except:
                            line = original_line
                            break
            
            # Reset prompt flag at end of string
            if in_prompt and (line.strip().endswith('"') or line.strip().endswith("'")):
                in_prompt = False
            
            mutated_lines.append(line)
        
        return '\n'.join(mutated_lines)
    
    def _mutate_logic_conditions(self, content: str) -> str:
        """Slightly modify logical conditions and comparisons"""
        
        # Mapping of condition mutations
        condition_mutations = {
            ' > ': ' >= ',
            ' < ': ' <= ',
            ' >= ': ' > ',
            ' <= ': ' < ',
            ' == ': ' != ',
            ' != ': ' == ',
            ' and ': ' or ',
            ' or ': ' and ',
            'True': 'False',
            'False': 'True'
        }
        
        if random.random() < 0.05:  # 5% chance (very conservative)
            for original, replacement in condition_mutations.items():
                if original in content and random.random() < 0.1:
                    # Only mutate one occurrence
                    content = content.replace(original, replacement, 1)
                    break
        
        return content
    
    def _mutate_variable_names(self, content: str) -> str:
        """Rename some variable names with variations"""
        
        variable_mutations = {
            'result': 'output',
            'output': 'result',
            'response': 'reply',
            'reply': 'response',
            'message': 'msg',
            'msg': 'message',
            'context': 'ctx',
            'ctx': 'context',
            'config': 'cfg',
            'cfg': 'config',
            'data': 'info',
            'info': 'data'
        }
        
        if random.random() < 0.1:  # 10% chance
            for original, replacement in variable_mutations.items():
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(original) + r'\b'
                if re.search(pattern, content) and random.random() < 0.3:
                    content = re.sub(pattern, replacement, content, count=1)
                    break
        
        return content
    
    def _mutate_control_flow(self, content: str) -> str:
        """Mutate control flow structures like if/else, loops"""
        
        lines = content.split('\n')
        mutated_lines = []
        
        for line in lines:
            # Mutate if conditions (rarely)
            if line.strip().startswith('if ') and random.random() < 0.02:  # Very conservative
                # Add simple condition checks
                if 'and' in line and random.random() < 0.5:
                    line = line.replace(' and ', ' or ', 1)
                elif 'or' in line and random.random() < 0.5:
                    line = line.replace(' or ', ' and ', 1)
            
            # Mutate return early patterns
            elif 'return None' in line and random.random() < 0.1:
                line = line.replace('return None', 'return False')
            elif 'return False' in line and random.random() < 0.1:
                line = line.replace('return False', 'return None')
            
            mutated_lines.append(line)
        
        return '\n'.join(mutated_lines)
    
    def _mutate_error_messages(self, content: str) -> str:
        """Mutate error messages and logging statements"""
        
        error_message_mutations = {
            'Error': 'Warning',
            'Failed': 'Unable',
            'Invalid': 'Incorrect',
            'Missing': 'Absent',
            'Unexpected': 'Unrecognized'
        }
        
        if random.random() < 0.1:  # 10% chance
            for original, replacement in error_message_mutations.items():
                if f'"{original}' in content and random.random() < 0.3:
                    content = content.replace(f'"{original}', f'"{replacement}', 1)
                    break
                elif f"'{original}" in content and random.random() < 0.3:
                    content = content.replace(f"'{original}", f"'{replacement}", 1)
                    break
        
        return content
    
    def _mutate_timeout_values(self, content: str) -> str:
        """Mutate timeout and delay values"""
        
        def mutate_timeout(match):
            num_str = match.group(1)
            try:
                num = int(num_str)
                if 5 <= num <= 300:  # Only mutate reasonable timeout values
                    variation = random.randint(-2, 2)
                    new_num = max(1, num + variation)
                    return f"timeout={new_num}"
            except:
                pass
            return match.group(0)
        
        # Mutate timeout parameters
        if random.random() < 0.1:  # 10% chance
            content = re.sub(r'timeout=(\d+)', mutate_timeout, content)
        
        return content
    
    def create_backup(self, file_path: str) -> Optional[str]:
        """Create a backup of the file before mutation"""
        try:
            file_path = Path(file_path)
            backup_path = file_path.parent / f"{file_path.stem}_backup{file_path.suffix}"
            
            shutil.copy2(file_path, backup_path)
            return str(backup_path)
            
        except Exception as e:
            logging.error(f"[MutationEngine] Failed to create backup: {e}")
            return None
    
    def restore_from_backup(self, file_path: str, backup_path: str) -> bool:
        """Restore file from backup"""
        try:
            shutil.copy2(backup_path, file_path)
            Path(backup_path).unlink()  # Remove backup after restore
            return True
            
        except Exception as e:
            logging.error(f"[MutationEngine] Failed to restore from backup: {e}")
            return False
    
    def mutate_file_with_rollback(self, file_path: str) -> bool:
        """Mutate file with automatic rollback on failure"""
        
        # Create backup first
        backup_path = self.create_backup(file_path)
        if not backup_path:
            logging.error(f"[MutationEngine] Cannot mutate {file_path}: backup failed")
            return False
        
        try:
            # Apply mutations
            success = self.mutate_file(file_path)
            
            if success:
                # Test that the mutated file still has valid syntax
                with open(file_path, 'r', encoding='utf-8') as f:
                    mutated_content = f.read()
                
                if not self.validate_syntax(mutated_content):
                    logging.warning(f"[MutationEngine] Mutation broke syntax, rolling back {file_path}")
                    self.restore_from_backup(file_path, backup_path)
                    return False
                
                # Clean up backup on success
                Path(backup_path).unlink()
                return True
            else:
                # Restore from backup on mutation failure
                self.restore_from_backup(file_path, backup_path)
                return False
                
        except Exception as e:
            logging.error(f"[MutationEngine] Error during mutation with rollback: {e}")
            # Restore from backup on any error
            self.restore_from_backup(file_path, backup_path)
            return False
    
    def validate_syntax(self, content: str) -> bool:
        """
        Validate that content has valid Python syntax
        
        Args:
            content: Python code content
            
        Returns:
            True if syntax is valid, False otherwise
        """
        
        try:
            ast.parse(content)
            return True
        except SyntaxError:
            return False
    
    def get_mutation_summary(self, original_content: str, mutated_content: str) -> Dict[str, Any]:
        """
        Generate a summary of mutations applied
        
        Args:
            original_content: Original file content
            mutated_content: Mutated file content
            
        Returns:
            Summary dictionary
        """
        
        original_lines = original_content.split('\n')
        mutated_lines = mutated_content.split('\n')
        
        changes = []
        for i, (orig, mut) in enumerate(zip(original_lines, mutated_lines)):
            if orig != mut:
                changes.append({
                    'line': i + 1,
                    'original': orig.strip(),
                    'mutated': mut.strip()
                })
        
        return {
            'total_lines': len(original_lines),
            'changed_lines': len(changes),
            'mutation_rate': len(changes) / len(original_lines) if original_lines else 0,
            'changes': changes[:10]  # Show first 10 changes
        }


class CodePatternMutator:
    """
    Advanced code pattern mutations for more sophisticated changes.
    """
    
    def __init__(self):
        self.pattern_mutations = [
            self._mutate_logging_statements,
            self._mutate_error_handling,
            self._mutate_return_statements,
            self._mutate_conditional_logic
        ]
    
    def apply_pattern_mutations(self, content: str, mutation_rate: float = 0.1) -> str:
        """Apply pattern-based mutations"""
        
        for pattern_mutation in self.pattern_mutations:
            if random.random() < mutation_rate:
                try:
                    content = pattern_mutation(content)
                except Exception as e:
                    logging.warning(f"[CodePatternMutator] Pattern mutation failed: {e}")
        
        return content
    
    def _mutate_logging_statements(self, content: str) -> str:
        """Modify logging statements"""
        
        logging_mutations = {
            'logging.info': 'logging.debug',
            'logging.debug': 'logging.info',
            'logging.warning': 'logging.error',
            'logging.error': 'logging.warning'
        }
        
        for original, replacement in logging_mutations.items():
            if original in content and random.random() < 0.3:
                content = content.replace(original, replacement, 1)
                break
        
        return content
    
    def _mutate_error_handling(self, content: str) -> str:
        """Modify error handling patterns"""
        
        lines = content.split('\n')
        mutated_lines = []
        
        for line in lines:
            # Add more specific exception handling
            if 'except Exception' in line and random.random() < 0.2:
                line = line.replace('except Exception', 'except (ValueError, TypeError, AttributeError)')
            
            # Modify exception messages
            elif 'raise Exception' in line and random.random() < 0.2:
                line = line.replace('raise Exception', 'raise ValueError')
            
            mutated_lines.append(line)
        
        return '\n'.join(mutated_lines)
    
    def _mutate_return_statements(self, content: str) -> str:
        """Modify return statement patterns"""
        
        # Add early returns or modify return values slightly
        return content  # Placeholder for more complex return mutations
    
    def _mutate_conditional_logic(self, content: str) -> str:
        """Modify conditional logic patterns"""
        
        # Modify if-else chains, add elif branches, etc.
        return content  # Placeholder for more complex conditional mutations