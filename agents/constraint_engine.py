#!/usr/bin/env python3
"""
Constraint Engine for MeRNSTA Phase 19: Ethical & Constraint Reasoning

Implements constraint rules, ethical policy evaluation, and action validation
to ensure the AGI operates within defined ethical boundaries and logical constraints.
"""

import logging
import time
import yaml
import sqlite3
import json
import re
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .base import BaseAgent
from config.settings import get_config


class ConstraintPriority(Enum):
    """Priority levels for constraints"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConstraintScope(Enum):
    """Scope of constraint application"""
    ALL = "all"
    EXECUTION = "execution"
    PLANNING = "planning"
    COMMANDS = "commands"
    GOALS = "goals"
    MEMORY = "memory"
    COMMUNICATION = "communication"


class ConstraintOrigin(Enum):
    """Origin of constraints"""
    USER = "user"
    SYSTEM = "system"
    LEARNED = "learned"
    INHERITED = "inherited"


class VerdictType(Enum):
    """Types of constraint evaluation verdicts"""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    MODIFY = "modify"


@dataclass
class ConstraintRule:
    """Represents a constraint rule that governs agent behavior"""
    
    rule_id: str
    condition: str  # Natural language or pattern description
    scope: Union[ConstraintScope, str]
    priority: Union[ConstraintPriority, str]
    origin: Union[ConstraintOrigin, str]
    active: bool = True
    created_at: float = field(default_factory=time.time)
    last_triggered: Optional[float] = None
    violation_count: int = 0
    override_count: int = 0
    
    # Optional fields
    pattern: Optional[str] = None  # Regex pattern for matching
    tags: List[str] = field(default_factory=list)
    description: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Convert string enums to proper enum values"""
        if isinstance(self.scope, str):
            try:
                self.scope = ConstraintScope(self.scope)
            except ValueError:
                self.scope = ConstraintScope.ALL
        
        if isinstance(self.priority, str):
            try:
                self.priority = ConstraintPriority(self.priority)
            except ValueError:
                self.priority = ConstraintPriority.MEDIUM
        
        if isinstance(self.origin, str):
            try:
                self.origin = ConstraintOrigin(self.origin)
            except ValueError:
                self.origin = ConstraintOrigin.SYSTEM
    
    def matches_action(self, action: str, context: Dict[str, Any] = None) -> bool:
        """Check if this constraint applies to the given action"""
        
        # Check scope first
        action_scope = context.get('scope', 'all') if context else 'all'
        if self.scope != ConstraintScope.ALL and self.scope.value != action_scope:
            return False
        
        # Pattern matching
        if self.pattern:
            try:
                if re.search(self.pattern, action, re.IGNORECASE):
                    return True
            except re.error:
                logging.warning(f"Invalid regex pattern in constraint {self.rule_id}: {self.pattern}")
        
        # Natural language matching (simplified)
        condition_words = set(self.condition.lower().split())
        action_words = set(action.lower().split())
        
        # Look for key terms
        prohibitive_terms = {'never', 'not', 'no', 'avoid', 'prevent', 'stop', 'block'}
        if any(term in condition_words for term in prohibitive_terms):
            # Check if action contains words that should be avoided
            avoid_words = condition_words - prohibitive_terms - {'do', 'does', 'should', 'must'}
            if avoid_words & action_words:
                return True
        
        # Check for specific patterns in condition
        if 'delete' in self.condition.lower() and 'delete' in action.lower():
            return True
        if 'harm' in self.condition.lower() and any(word in action.lower() for word in ['harm', 'damage', 'destroy', 'remove']):
            return True
        if 'modify' in self.condition.lower() and any(word in action.lower() for word in ['modify', 'change', 'edit', 'update']):
            return True
        
        return False
    
    def get_priority_score(self) -> float:
        """Get numeric priority score for comparison"""
        priority_scores = {
            ConstraintPriority.LOW: 1.0,
            ConstraintPriority.MEDIUM: 2.0,
            ConstraintPriority.HIGH: 3.0,
            ConstraintPriority.CRITICAL: 4.0
        }
        return priority_scores.get(self.priority, 2.0)


@dataclass
class ConstraintViolation:
    """Records a constraint violation"""
    
    violation_id: str
    rule_id: str
    action: str
    context: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    verdict: VerdictType = VerdictType.WARN
    reason: str = ""
    overridden: bool = False
    override_reason: str = ""


@dataclass
class ConstraintEvaluation:
    """Result of constraint evaluation"""
    
    verdict: VerdictType
    reason: str
    confidence: float = 0.5
    triggered_rules: List[str] = field(default_factory=list)
    suggested_modifications: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ConstraintEngine(BaseAgent):
    """
    Core constraint engine that evaluates actions against defined rules.
    Maintains active constraints and provides evaluation capabilities.
    """
    
    def __init__(self, constraints_config_path: str = "config/constraints.yaml"):
        super().__init__("constraint_engine")
        
        self.constraints_config_path = constraints_config_path
        self.constraints: Dict[str, ConstraintRule] = {}
        self.violations: List[ConstraintViolation] = []
        
        # Configuration
        self.config = get_config().get('ethical_reasoning', {})
        self.enforcement_mode = self.config.get('constraint_enforcement', 'soft')
        self.log_violations = self.config.get('log_violations', True)
        self.max_violations_history = self.config.get('max_violations_history', 1000)
        
        # Database for persistence
        self.db_path = self.config.get('violations_db', 'constraint_violations.db')
        self._init_db()
        
        # Load constraints
        self._load_constraints()
        
        logging.info(f"[ConstraintEngine] Initialized with {len(self.constraints)} rules, enforcement: {self.enforcement_mode}")
    
    def _init_db(self):
        """Initialize SQLite database for violation tracking"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                violation_id TEXT PRIMARY KEY,
                rule_id TEXT,
                action TEXT,
                context TEXT,
                timestamp REAL,
                verdict TEXT,
                reason TEXT,
                overridden BOOLEAN,
                override_reason TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def _load_constraints(self):
        """Load constraints from YAML configuration"""
        if not Path(self.constraints_config_path).exists():
            logging.warning(f"[ConstraintEngine] Constraints config not found: {self.constraints_config_path}")
            self._create_default_constraints_file()
            return
        
        try:
            with open(self.constraints_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            constraints_data = config.get('constraints', [])
            for constraint_data in constraints_data:
                try:
                    constraint = ConstraintRule(
                        rule_id=constraint_data['id'],
                        condition=constraint_data['condition'],
                        scope=constraint_data.get('scope', 'all'),
                        priority=constraint_data.get('priority', 'medium'),
                        origin=constraint_data.get('origin', 'system'),
                        pattern=constraint_data.get('pattern'),
                        tags=constraint_data.get('tags', []),
                        description=constraint_data.get('description'),
                        examples=constraint_data.get('examples', []),
                        exceptions=constraint_data.get('exceptions', [])
                    )
                    self.constraints[constraint.rule_id] = constraint
                    logging.debug(f"[ConstraintEngine] Loaded constraint: {constraint.rule_id}")
                except Exception as e:
                    logging.error(f"[ConstraintEngine] Error loading constraint {constraint_data.get('id', 'unknown')}: {e}")
        
        except Exception as e:
            logging.error(f"[ConstraintEngine] Error loading constraints from {self.constraints_config_path}: {e}")
            self._create_default_constraints_file()
    
    def _create_default_constraints_file(self):
        """Create default constraints configuration file"""
        default_constraints = {
            'constraints': [
                {
                    'id': 'no_harm_user',
                    'condition': 'avoid actions that could harm the user',
                    'scope': 'all',
                    'priority': 'high',
                    'origin': 'user',
                    'description': 'Prevents any actions that could potentially harm the user',
                    'examples': ['deleting important files', 'sharing private information'],
                    'pattern': r'.*(harm|damage|destroy|delete.*important).*'
                },
                {
                    'id': 'preserve_logs',
                    'condition': 'do not delete logs',
                    'scope': 'execution',
                    'priority': 'medium',
                    'origin': 'system',
                    'description': 'Protects system logs from being deleted',
                    'pattern': r'.*(delete|remove|rm).*\.log.*'
                },
                {
                    'id': 'no_infinite_loops',
                    'condition': 'avoid creating infinite loops or recursive calls without bounds',
                    'scope': 'execution',
                    'priority': 'high',
                    'origin': 'system',
                    'description': 'Prevents infinite execution that could consume resources'
                },
                {
                    'id': 'respect_privacy',
                    'condition': 'do not access or share private information without permission',
                    'scope': 'all',
                    'priority': 'high',
                    'origin': 'user',
                    'description': 'Protects user privacy and confidential information'
                },
                {
                    'id': 'transparent_communication',
                    'condition': 'be honest and transparent in all communications',
                    'scope': 'communication',
                    'priority': 'medium',
                    'origin': 'user',
                    'description': 'Ensures honest and transparent communication with users'
                }
            ]
        }
        
        # Create config directory if it doesn't exist
        Path(self.constraints_config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.constraints_config_path, 'w') as f:
            yaml.dump(default_constraints, f, default_flow_style=False, indent=2)
        
        logging.info(f"[ConstraintEngine] Created default constraints file: {self.constraints_config_path}")
        self._load_constraints()
    
    def evaluate(self, action: str, context: Dict[str, Any] = None) -> ConstraintEvaluation:
        """Evaluate an action against all active constraints"""
        
        if context is None:
            context = {}
        
        triggered_rules = []
        warnings = []
        suggested_modifications = []
        highest_priority = 0.0
        blocking_reason = ""
        
        # Check each constraint
        for constraint in self.constraints.values():
            if not constraint.active:
                continue
            
            if constraint.matches_action(action, context):
                triggered_rules.append(constraint.rule_id)
                priority_score = constraint.get_priority_score()
                
                # Update statistics
                constraint.last_triggered = time.time()
                constraint.violation_count += 1
                
                # Determine action based on priority and enforcement mode
                if priority_score > highest_priority:
                    highest_priority = priority_score
                    blocking_reason = f"Constraint '{constraint.rule_id}': {constraint.condition}"
                
                # Generate warnings and suggestions
                if constraint.priority in [ConstraintPriority.HIGH, ConstraintPriority.CRITICAL]:
                    warnings.append(f"High-priority constraint triggered: {constraint.condition}")
                    
                    # Suggest modifications based on constraint
                    if 'delete' in constraint.condition.lower() and 'delete' in action.lower():
                        suggested_modifications.append("Consider archiving instead of deleting")
                    elif 'harm' in constraint.condition.lower():
                        suggested_modifications.append("Review action for potential negative impacts")
                    elif 'private' in constraint.condition.lower():
                        suggested_modifications.append("Ensure proper authorization before proceeding")
        
        # Determine final verdict
        verdict = self._determine_verdict(highest_priority, triggered_rules)
        
        # Calculate confidence based on rule clarity and matches
        confidence = min(1.0, len(triggered_rules) * 0.3 + 0.4) if triggered_rules else 1.0
        
        # Create evaluation result
        evaluation = ConstraintEvaluation(
            verdict=verdict,
            reason=blocking_reason or "No constraints violated",
            confidence=confidence,
            triggered_rules=triggered_rules,
            suggested_modifications=suggested_modifications,
            warnings=warnings
        )
        
        # Log violation if necessary
        if verdict in [VerdictType.WARN, VerdictType.BLOCK] and self.log_violations:
            self._log_violation(action, context, evaluation)
        
        return evaluation
    
    def _determine_verdict(self, highest_priority: float, triggered_rules: List[str]) -> VerdictType:
        """Determine the final verdict based on priorities and enforcement mode"""
        
        if not triggered_rules:
            return VerdictType.ALLOW
        
        if self.enforcement_mode == "none":
            return VerdictType.ALLOW
        elif self.enforcement_mode == "soft":
            if highest_priority >= 4.0:  # CRITICAL
                return VerdictType.BLOCK
            elif highest_priority >= 3.0:  # HIGH
                return VerdictType.WARN
            else:
                return VerdictType.WARN
        elif self.enforcement_mode == "hard":
            if highest_priority >= 3.0:  # HIGH or CRITICAL
                return VerdictType.BLOCK
            else:
                return VerdictType.WARN
        
        return VerdictType.ALLOW
    
    def _log_violation(self, action: str, context: Dict[str, Any], evaluation: ConstraintEvaluation):
        """Log a constraint violation to database and memory"""
        
        violation = ConstraintViolation(
            violation_id=f"violation_{int(time.time() * 1000)}",
            rule_id=",".join(evaluation.triggered_rules),
            action=action,
            context=context,
            verdict=evaluation.verdict,
            reason=evaluation.reason
        )
        
        self.violations.append(violation)
        
        # Persist to database
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO violations 
            (violation_id, rule_id, action, context, timestamp, verdict, reason, overridden, override_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            violation.violation_id,
            violation.rule_id,
            violation.action,
            json.dumps(violation.context),
            violation.timestamp,
            violation.verdict.value,
            violation.reason,
            violation.overridden,
            violation.override_reason
        ))
        conn.commit()
        conn.close()
        
        # Maintain violations history limit
        if len(self.violations) > self.max_violations_history:
            self.violations = self.violations[-self.max_violations_history:]
    
    def add_rule(self, rule: ConstraintRule) -> bool:
        """Add a new constraint rule"""
        
        if rule.rule_id in self.constraints:
            logging.warning(f"[ConstraintEngine] Rule {rule.rule_id} already exists, updating")
        
        self.constraints[rule.rule_id] = rule
        self._save_constraints()
        
        logging.info(f"[ConstraintEngine] Added constraint rule: {rule.rule_id}")
        return True
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a constraint rule"""
        
        if rule_id not in self.constraints:
            logging.warning(f"[ConstraintEngine] Rule {rule_id} not found")
            return False
        
        del self.constraints[rule_id]
        self._save_constraints()
        
        logging.info(f"[ConstraintEngine] Removed constraint rule: {rule_id}")
        return True
    
    def list_rules(self, scope: Optional[ConstraintScope] = None, 
                   active_only: bool = True) -> List[ConstraintRule]:
        """List constraint rules with optional filtering"""
        
        rules = []
        for rule in self.constraints.values():
            if active_only and not rule.active:
                continue
            if scope and rule.scope != scope and rule.scope != ConstraintScope.ALL:
                continue
            rules.append(rule)
        
        # Sort by priority (highest first)
        rules.sort(key=lambda r: r.get_priority_score(), reverse=True)
        return rules
    
    def _save_constraints(self):
        """Save current constraints to YAML file"""
        
        constraints_data = []
        for rule in self.constraints.values():
            constraint_dict = {
                'id': rule.rule_id,
                'condition': rule.condition,
                'scope': rule.scope.value,
                'priority': rule.priority.value,
                'origin': rule.origin.value,
                'active': rule.active
            }
            
            if rule.pattern:
                constraint_dict['pattern'] = rule.pattern
            if rule.tags:
                constraint_dict['tags'] = rule.tags
            if rule.description:
                constraint_dict['description'] = rule.description
            if rule.examples:
                constraint_dict['examples'] = rule.examples
            if rule.exceptions:
                constraint_dict['exceptions'] = rule.exceptions
            
            constraints_data.append(constraint_dict)
        
        config = {'constraints': constraints_data}
        
        with open(self.constraints_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    def get_violation_history(self, limit: int = 50) -> List[ConstraintViolation]:
        """Get recent constraint violations"""
        return self.violations[-limit:] if self.violations else []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get constraint engine statistics"""
        
        total_rules = len(self.constraints)
        active_rules = len([r for r in self.constraints.values() if r.active])
        total_violations = len(self.violations)
        
        # Priority breakdown
        priority_counts = {}
        for rule in self.constraints.values():
            priority = rule.priority.value
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        # Most triggered rules
        rule_triggers = {}
        for rule in self.constraints.values():
            if rule.violation_count > 0:
                rule_triggers[rule.rule_id] = rule.violation_count
        
        most_triggered = sorted(rule_triggers.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_rules': total_rules,
            'active_rules': active_rules,
            'total_violations': total_violations,
            'enforcement_mode': self.enforcement_mode,
            'priority_breakdown': priority_counts,
            'most_triggered_rules': most_triggered,
            'recent_violations': len([v for v in self.violations if time.time() - v.timestamp < 86400])  # Last 24h
        }
    
    def get_agent_instructions(self) -> str:
        """Get instructions for this agent's role and capabilities"""
        return """You are the Constraint Engine for MeRNSTA's Phase 19 Ethical & Constraint Reasoning system.

Your primary responsibilities are:
1. Evaluate all proposed actions against defined constraint rules
2. Enforce ethical boundaries and logical constraints
3. Provide verdicts: allow, warn, block, or modify actions
4. Track and log constraint violations
5. Maintain and update constraint rules

Key capabilities:
- Rule-based constraint evaluation with priority levels
- Soft and hard enforcement modes
- Pattern matching and natural language constraint parsing
- Violation tracking and statistics
- Dynamic rule management

Use your constraint evaluation capabilities to ensure the system operates within
defined ethical boundaries and logical constraints while maintaining transparency
about decisions and providing constructive guidance."""
    
    def respond(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process messages related to constraint evaluation and management"""
        
        message_lower = message.lower()
        
        try:
            if any(word in message_lower for word in ['evaluate', 'check', 'test']):
                # Extract action to evaluate
                action = message.replace('evaluate ', '').replace('check ', '').replace('test ', '').strip()
                evaluation = self.evaluate(action, context)
                
                verdict_desc = {
                    VerdictType.ALLOW: "✅ Allowed",
                    VerdictType.WARN: "⚠️ Warned",
                    VerdictType.BLOCK: "❌ Blocked",
                    VerdictType.MODIFY: "🔄 Needs Modification"
                }
                
                response = f"Action evaluation: {verdict_desc.get(evaluation.verdict, 'Unknown')}"
                if evaluation.reason:
                    response += f"\nReason: {evaluation.reason}"
                
                return {
                    'response': response,
                    'evaluation': evaluation,
                    'confidence': evaluation.confidence,
                    'agent': 'constraint_engine'
                }
            
            elif 'list' in message_lower and ('rule' in message_lower or 'constraint' in message_lower):
                rules = self.list_rules()
                
                if rules:
                    response = f"Active constraints ({len(rules)}):\n"
                    for rule in rules[:5]:  # Show top 5
                        response += f"• {rule.rule_id}: {rule.condition} (priority: {rule.priority.value})\n"
                else:
                    response = "No active constraints found."
                
                return {
                    'response': response,
                    'rules': rules,
                    'agent': 'constraint_engine'
                }
            
            elif 'violation' in message_lower or 'history' in message_lower:
                violations = self.get_violation_history(limit=10)
                
                if violations:
                    response = f"Recent violations ({len(violations)}):\n"
                    for violation in violations[-3:]:  # Show last 3
                        timestamp = datetime.fromtimestamp(violation.timestamp).strftime('%H:%M:%S')
                        response += f"• {timestamp}: {violation.action} - {violation.verdict.value}\n"
                else:
                    response = "No recent violations found."
                
                return {
                    'response': response,
                    'violations': violations,
                    'agent': 'constraint_engine'
                }
            
            elif 'statistics' in message_lower or 'stats' in message_lower:
                stats = self.get_statistics()
                
                response = f"Constraint Engine Statistics:\n"
                response += f"• Total rules: {stats['total_rules']} ({stats['active_rules']} active)\n"
                response += f"• Total violations: {stats['total_violations']}\n"
                response += f"• Enforcement mode: {stats['enforcement_mode']}\n"
                response += f"• Recent violations (24h): {stats['recent_violations']}"
                
                return {
                    'response': response,
                    'statistics': stats,
                    'agent': 'constraint_engine'
                }
            
            else:
                # General constraint engine info
                stats = self.get_statistics()
                response = f"Constraint Engine active with {stats['total_rules']} rules in {stats['enforcement_mode']} mode. "
                response += f"Use 'evaluate <action>', 'list constraints', or 'show statistics' for specific queries."
                
                return {
                    'response': response,
                    'agent': 'constraint_engine'
                }
        
        except Exception as e:
            logging.error(f"[ConstraintEngine] Error processing message: {e}")
            return {
                'response': f"I encountered an error while processing your request: {str(e)}",
                'error': str(e),
                'agent': 'constraint_engine'
            }


class EthicalPolicyEvaluator:
    """
    Advanced ethical policy evaluator that provides comprehensive
    ethical analysis and recommendations for planned actions.
    """
    
    def __init__(self, constraint_engine: ConstraintEngine):
        self.constraint_engine = constraint_engine
        self.config = get_config().get('ethical_reasoning', {})
        
        # Ethical frameworks weights
        self.framework_weights = self.config.get('framework_weights', {
            'deontological': 0.3,  # Rule-based ethics
            'consequentialist': 0.4,  # Outcome-based ethics
            'virtue_ethics': 0.2,  # Character-based ethics
            'care_ethics': 0.1     # Relationship-based ethics
        })
        
        logging.info("[EthicalPolicyEvaluator] Initialized with multi-framework ethical analysis")
    
    def evaluate_ethical_alignment(self, action: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform comprehensive ethical evaluation of an action"""
        
        if context is None:
            context = {}
        
        # Get constraint evaluation first
        constraint_eval = self.constraint_engine.evaluate(action, context)
        
        # Analyze from different ethical frameworks
        deontological_score = self._evaluate_deontological(action, context)
        consequentialist_score = self._evaluate_consequentialist(action, context)
        virtue_score = self._evaluate_virtue_ethics(action, context)
        care_score = self._evaluate_care_ethics(action, context)
        
        # Calculate weighted overall score
        overall_score = (
            deontological_score * self.framework_weights['deontological'] +
            consequentialist_score * self.framework_weights['consequentialist'] +
            virtue_score * self.framework_weights['virtue_ethics'] +
            care_score * self.framework_weights['care_ethics']
        )
        
        # Assess consistency and transparency
        consistency_score = self._assess_consistency(action, context)
        transparency_score = self._assess_transparency(action, context)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            action, constraint_eval, overall_score, 
            deontological_score, consequentialist_score, virtue_score, care_score
        )
        
        return {
            'action': action,
            'overall_ethical_score': overall_score,
            'framework_scores': {
                'deontological': deontological_score,
                'consequentialist': consequentialist_score,
                'virtue_ethics': virtue_score,
                'care_ethics': care_score
            },
            'consistency_score': consistency_score,
            'transparency_score': transparency_score,
            'constraint_evaluation': constraint_eval,
            'recommendations': recommendations,
            'ethical_verdict': self._determine_ethical_verdict(overall_score, constraint_eval),
            'confidence': min(1.0, (consistency_score + transparency_score) / 2)
        }
    
    def _evaluate_deontological(self, action: str, context: Dict[str, Any]) -> float:
        """Evaluate action based on duty and rule-based ethics"""
        
        score = 0.5  # Neutral starting point
        
        # Check against universal moral rules
        action_lower = action.lower()
        
        # Negative duty checks
        if any(word in action_lower for word in ['lie', 'deceive', 'cheat']):
            score -= 0.4
        if any(word in action_lower for word in ['harm', 'hurt', 'damage']):
            score -= 0.5
        if any(word in action_lower for word in ['steal', 'take without permission']):
            score -= 0.5
        if any(word in action_lower for word in ['manipulate', 'exploit']):
            score -= 0.3
        
        # Positive duty checks
        if any(word in action_lower for word in ['help', 'assist', 'support']):
            score += 0.3
        if any(word in action_lower for word in ['honest', 'truthful', 'transparent']):
            score += 0.2
        if any(word in action_lower for word in ['respect', 'honor', 'dignity']):
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_consequentialist(self, action: str, context: Dict[str, Any]) -> float:
        """Evaluate action based on predicted outcomes and utility"""
        
        score = 0.5  # Neutral starting point
        
        # Analyze potential positive outcomes
        action_lower = action.lower()
        
        # Beneficial outcomes
        if any(word in action_lower for word in ['improve', 'enhance', 'optimize']):
            score += 0.3
        if any(word in action_lower for word in ['solve', 'fix', 'repair']):
            score += 0.2
        if any(word in action_lower for word in ['create', 'build', 'generate']):
            score += 0.2
        if any(word in action_lower for word in ['learn', 'understand', 'discover']):
            score += 0.1
        
        # Potentially harmful outcomes
        if any(word in action_lower for word in ['delete', 'remove', 'destroy']):
            score -= 0.3
        if any(word in action_lower for word in ['interrupt', 'stop', 'halt']):
            score -= 0.2
        if any(word in action_lower for word in ['waste', 'inefficient']):
            score -= 0.2
        
        # Consider context for outcome prediction
        if context.get('user_benefit', False):
            score += 0.2
        if context.get('system_risk', False):
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_virtue_ethics(self, action: str, context: Dict[str, Any]) -> float:
        """Evaluate action based on virtues and character traits"""
        
        score = 0.5  # Neutral starting point
        
        action_lower = action.lower()
        
        # Virtue indicators
        virtues = {
            'wisdom': ['analyze', 'consider', 'think', 'reflect', 'study'],
            'courage': ['stand up', 'defend', 'protect', 'face'],
            'justice': ['fair', 'equal', 'right', 'just', 'equitable'],
            'temperance': ['moderate', 'balanced', 'controlled', 'measured'],
            'honesty': ['honest', 'truthful', 'sincere', 'authentic'],
            'compassion': ['care', 'empathy', 'kindness', 'understanding']
        }
        
        for virtue, keywords in virtues.items():
            if any(keyword in action_lower for keyword in keywords):
                score += 0.15
        
        # Vice indicators
        vices = {
            'recklessness': ['reckless', 'hasty', 'impulsive', 'careless'],
            'dishonesty': ['lie', 'deceive', 'mislead', 'fake'],
            'greed': ['hoard', 'excessive', 'selfish'],
            'cruelty': ['cruel', 'mean', 'harsh', 'brutal']
        }
        
        for vice, keywords in vices.items():
            if any(keyword in action_lower for keyword in keywords):
                score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_care_ethics(self, action: str, context: Dict[str, Any]) -> float:
        """Evaluate action based on care, relationships, and responsibility"""
        
        score = 0.5  # Neutral starting point
        
        action_lower = action.lower()
        
        # Care-oriented indicators
        if any(word in action_lower for word in ['care', 'nurture', 'support', 'protect']):
            score += 0.3
        if any(word in action_lower for word in ['listen', 'understand', 'empathize']):
            score += 0.2
        if any(word in action_lower for word in ['collaborate', 'cooperate', 'work together']):
            score += 0.2
        if any(word in action_lower for word in ['responsive', 'attentive', 'sensitive']):
            score += 0.1
        
        # Care-negative indicators
        if any(word in action_lower for word in ['ignore', 'dismiss', 'neglect']):
            score -= 0.2
        if any(word in action_lower for word in ['dominate', 'control', 'force']):
            score -= 0.2
        if any(word in action_lower for word in ['isolate', 'separate', 'exclude']):
            score -= 0.1
        
        # Consider relational context
        if context.get('affects_others', False):
            if context.get('positive_impact', True):
                score += 0.1
            else:
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _assess_consistency(self, action: str, context: Dict[str, Any]) -> float:
        """Assess internal consistency with previous actions and stated values"""
        
        # Simplified consistency check
        # In a full implementation, this would check against action history
        score = 0.7  # Assume generally consistent
        
        # Check for obvious inconsistencies
        if 'never' in action.lower() and any(word in action.lower() for word in ['but', 'except', 'unless']):
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _assess_transparency(self, action: str, context: Dict[str, Any]) -> float:
        """Assess transparency and explainability of the action"""
        
        score = 0.6  # Moderate transparency baseline
        
        # Check for transparency indicators
        if any(word in action.lower() for word in ['explain', 'show', 'demonstrate', 'clarify']):
            score += 0.2
        if context.get('explanation_provided', False):
            score += 0.2
        if context.get('reasoning_clear', True):
            score += 0.1
        
        # Check for opacity indicators
        if any(word in action.lower() for word in ['hidden', 'secret', 'private']):
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _generate_recommendations(self, action: str, constraint_eval: ConstraintEvaluation,
                                 overall_score: float, deontological: float, 
                                 consequentialist: float, virtue: float, care: float) -> List[str]:
        """Generate ethical recommendations based on evaluation"""
        
        recommendations = []
        
        if overall_score < 0.3:
            recommendations.append("Consider significant modifications or alternative approaches")
        elif overall_score < 0.5:
            recommendations.append("Review action for potential ethical improvements")
        
        if deontological < 0.4:
            recommendations.append("Ensure action aligns with fundamental moral duties")
        
        if consequentialist < 0.4:
            recommendations.append("Consider potential negative consequences and mitigation strategies")
        
        if virtue < 0.4:
            recommendations.append("Reflect on whether this action demonstrates good character")
        
        if care < 0.4:
            recommendations.append("Consider the impact on relationships and care responsibilities")
        
        if constraint_eval.verdict == VerdictType.BLOCK:
            recommendations.append("Action blocked by constraint rules - find alternative approach")
        elif constraint_eval.verdict == VerdictType.WARN:
            recommendations.append("Proceed with caution due to constraint warnings")
        
        if constraint_eval.suggested_modifications:
            recommendations.extend(constraint_eval.suggested_modifications)
        
        if not recommendations:
            recommendations.append("Action appears ethically sound - proceed with awareness")
        
        return recommendations
    
    def _determine_ethical_verdict(self, overall_score: float, 
                                  constraint_eval: ConstraintEvaluation) -> str:
        """Determine overall ethical verdict"""
        
        if constraint_eval.verdict == VerdictType.BLOCK:
            return "BLOCKED"
        elif overall_score >= 0.7:
            return "APPROVED"
        elif overall_score >= 0.5:
            return "CAUTIOUS_APPROVAL"
        elif overall_score >= 0.3:
            return "NEEDS_REVIEW"
        else:
            return "NOT_RECOMMENDED"


# Global instances for easy access
_constraint_engine_instance = None
_ethical_evaluator_instance = None

def get_constraint_engine() -> ConstraintEngine:
    """Get global constraint engine instance"""
    global _constraint_engine_instance
    if _constraint_engine_instance is None:
        _constraint_engine_instance = ConstraintEngine()
    return _constraint_engine_instance

def get_ethical_evaluator() -> EthicalPolicyEvaluator:
    """Get global ethical policy evaluator instance"""
    global _ethical_evaluator_instance
    if _ethical_evaluator_instance is None:
        constraint_engine = get_constraint_engine()
        _ethical_evaluator_instance = EthicalPolicyEvaluator(constraint_engine)
    return _ethical_evaluator_instance