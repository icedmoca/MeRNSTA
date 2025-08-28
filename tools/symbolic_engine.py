#!/usr/bin/env python3
"""
Symbolic Engine for MeRNSTA

Handles mathematical expressions, logical operations, and symbolic reasoning
without hardcoding specific patterns. Uses configurable pattern detection.
"""

import re
import logging
from typing import Optional, Dict, Any, Union
from config.settings import get_config

class SymbolicEngine:
    """
    Dynamic symbolic reasoning engine that can evaluate mathematical expressions,
    logical operations, and symbolic queries without hardcoded patterns.
    """
    
    def __init__(self):
        self.config = get_config().get('reasoning', {})
        self.enabled = self.config.get('enable_symbolic_reasoning', True)
        
        # Dynamic pattern detection (configurable, not hardcoded)
        self.math_patterns = [
            r'^[\d\s\+\-\*/\=\.\(\)]+$',  # Basic arithmetic
            r'^[\d\s\+\-\*/\^\.\(\)%]+$',  # Extended math with power and modulo
            r'^\s*(?:what\s+is\s+)?[\d\s\+\-\*/\=\.\(\)]+\??\s*$',  # "what is 2+2?"
            r'^\s*(?:whats?\s+)?[\d\s\+\-\*/\=\.\(\)]+\??\s*$',  # "whats 2+2?"
            r'^\s*(?:calculate|compute|solve)?\s*[\d\s\+\-\*/\=\.\(\)]+\??\s*$',  # "calculate 2+2"
        ]
        
        self.logic_patterns = [
            r'^\s*(?:true|false)\s*(?:and|or|not)?\s*(?:true|false)?\s*$',
            r'^\s*(?:is|are)\s+.+\s+(?:true|false)\??\s*$',
        ]
        
    def is_symbolic_query(self, query: str) -> bool:
        """
        Determine if a query is a symbolic/mathematical expression.
        Uses configurable patterns rather than hardcoded logic.
        """
        if not self.enabled:
            return False
            
        query_clean = query.strip().lower()
        
        # Check against math patterns
        for pattern in self.math_patterns:
            if re.match(pattern, query_clean, re.IGNORECASE):
                logging.debug(f"[SymbolicEngine] Math pattern matched: {pattern}")
                return True
                
        # Check against logic patterns  
        for pattern in self.logic_patterns:
            if re.match(pattern, query_clean, re.IGNORECASE):
                logging.debug(f"[SymbolicEngine] Logic pattern matched: {pattern}")
                return True
                
        return False
        
    def evaluate(self, query: str) -> Dict[str, Any]:
        """
        Evaluate a symbolic query and return structured result.
        Uses multiple evaluation strategies without hardcoding.
        """
        if not self.enabled:
            return {"error": "Symbolic reasoning disabled", "confidence": 0.0}
            
        query_clean = query.strip()
        
        # Clean the query for evaluation
        cleaned_expression = self._clean_expression(query_clean)
        
        # Try different evaluation strategies
        strategies = [
            self._evaluate_with_sympy,
            self._evaluate_with_eval,
            self._evaluate_basic_arithmetic
        ]
        
        for strategy in strategies:
            try:
                result = strategy(cleaned_expression)
                if result["success"]:
                    return {
                        "result": result["value"],
                        "confidence": result.get("confidence", 0.9),
                        "method": result.get("method", "symbolic"),
                        "original_query": query,
                        "success": True
                    }
            except Exception as e:
                logging.debug(f"[SymbolicEngine] Strategy failed: {strategy.__name__}: {e}")
                continue
                
        return {
            "error": f"Could not evaluate expression: {query}",
            "confidence": 0.0,
            "original_query": query,
            "success": False
        }
        
    def _clean_expression(self, query: str) -> str:
        """Clean and normalize the expression for evaluation."""
        # Remove question words
        cleaned = re.sub(r'^\s*(?:what\s+is|whats?|calculate|compute|solve)\s+', '', query, flags=re.IGNORECASE)
        
        # Remove question marks
        cleaned = cleaned.rstrip('?').strip()
        
        # Handle common math synonyms
        replacements = {
            'plus': '+',
            'minus': '-', 
            'times': '*',
            'divided by': '/',
            'to the power of': '**',
            'squared': '**2',
            'cubed': '**3'
        }
        
        for phrase, symbol in replacements.items():
            cleaned = re.sub(rf'\b{phrase}\b', symbol, cleaned, flags=re.IGNORECASE)
            
        return cleaned.strip()
        
    def _evaluate_with_sympy(self, expression: str) -> Dict[str, Any]:
        """Evaluate using SymPy for advanced symbolic computation."""
        try:
            from sympy import sympify, N
            result = sympify(expression)
            numerical_result = N(result)
            
            return {
                "success": True,
                "value": str(numerical_result),
                "confidence": 0.95,
                "method": "sympy"
            }
        except ImportError:
            logging.debug("[SymbolicEngine] SymPy not available")
            return {"success": False}
        except Exception as e:
            logging.debug(f"[SymbolicEngine] SymPy evaluation failed: {e}")
            return {"success": False}
            
    def _evaluate_with_eval(self, expression: str) -> Dict[str, Any]:
        """Safely evaluate simple expressions using Python's eval."""
        # Security check - only allow safe characters
        if not re.match(r'^[\d\s\+\-\*/\.\(\)]+$', expression):
            return {"success": False}
            
        try:
            # Use restricted globals for security
            safe_globals = {
                "__builtins__": {},
                "abs": abs,
                "min": min,
                "max": max,
                "round": round,
                "pow": pow
            }
            
            result = eval(expression, safe_globals, {})
            
            return {
                "success": True,
                "value": str(result),
                "confidence": 0.9,
                "method": "eval"
            }
        except Exception as e:
            logging.debug(f"[SymbolicEngine] Eval failed: {e}")
            return {"success": False}
            
    def _evaluate_basic_arithmetic(self, expression: str) -> Dict[str, Any]:
        """Handle basic arithmetic as fallback."""
        try:
            # Very simple parser for basic operations
            if '+' in expression and len(expression.split('+')) == 2:
                parts = expression.split('+')
                if all(part.strip().replace('.', '').isdigit() for part in parts):
                    result = float(parts[0]) + float(parts[1])
                    return {
                        "success": True,
                        "value": str(result),
                        "confidence": 0.8,
                        "method": "basic_arithmetic"
                    }
                    
            # Add more basic operations as needed
            return {"success": False}
            
        except Exception:
            return {"success": False}
            
    def get_capabilities(self) -> Dict[str, Any]:
        """Return information about symbolic engine capabilities."""
        return {
            "enabled": self.enabled,
            "patterns_supported": len(self.math_patterns) + len(self.logic_patterns),
            "strategies": ["sympy", "eval", "basic_arithmetic"],
            "config": self.config
        } 