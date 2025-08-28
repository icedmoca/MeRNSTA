#!/usr/bin/env python3
"""
Agent Contract System for MeRNSTA Multi-Agent Cognitive System

Provides declarative role specification and task alignment scoring for each agent.
Each agent declares their core purpose, capabilities, and operating conditions.
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Union
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentContract:
    """
    Declarative contract specifying an agent's role, capabilities, and task preferences.
    
    Used for:
    - Intelligent task routing based on agent fitness
    - Agent self-evaluation and performance tracking
    - Capability drift detection over time
    - Role specialization and optimization
    """
    
    agent_name: str
    purpose: str
    capabilities: List[str] = field(default_factory=list)
    preferred_tasks: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    confidence_vector: Dict[str, float] = field(default_factory=dict)
    version: str = "1.0.0"
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Additional metadata for tracking
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    specialization_drift: Dict[str, float] = field(default_factory=dict)
    ideal_conditions: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default confidence vector if not provided."""
        if not self.confidence_vector:
            self.confidence_vector = self._get_default_confidence_vector()
    
    def _get_default_confidence_vector(self) -> Dict[str, float]:
        """Get default confidence scores based on agent type."""
        defaults = {
            'planning': 0.5,
            'reasoning': 0.5,
            'analysis': 0.5,
            'creativity': 0.5,
            'execution': 0.5,
            'debugging': 0.5,
            'optimization': 0.5,
            'communication': 0.5,
            'problem_solving': 0.5,
            'domain_expertise': 0.5
        }
        
        # Agent-specific default adjustments based on memory [[memory:4199483]]
        agent_specializations = {
            'planner': {'planning': 0.9, 'reasoning': 0.8, 'execution': 0.6},
            'critic': {'analysis': 0.9, 'reasoning': 0.8, 'problem_solving': 0.7},
            'debater': {'communication': 0.9, 'reasoning': 0.8, 'analysis': 0.7},
            'reflector': {'analysis': 0.8, 'reasoning': 0.9, 'problem_solving': 0.7},
            'architect': {'planning': 0.8, 'domain_expertise': 0.9, 'optimization': 0.8},
            'code_refactorer': {'debugging': 0.9, 'optimization': 0.8, 'execution': 0.8},
            'world_modeler': {'domain_expertise': 0.9, 'reasoning': 0.8, 'analysis': 0.8},
            'constraint_engine': {'optimization': 0.9, 'reasoning': 0.8, 'problem_solving': 0.8},
            'self_healer': {'debugging': 0.9, 'analysis': 0.8, 'problem_solving': 0.9},
            'recursive_planner': {'planning': 0.9, 'reasoning': 0.9, 'execution': 0.7},
            'self_prompter': {'creativity': 0.9, 'communication': 0.8, 'reasoning': 0.7}
        }
        
        if self.agent_name in agent_specializations:
            defaults.update(agent_specializations[self.agent_name])
        
        return defaults
    
    def score_alignment(self, task: Union[str, Dict[str, Any]]) -> float:
        """
        Score how well this agent aligns with a given task.
        
        Args:
            task: Task description (string) or task object with metadata
            
        Returns:
            Alignment score between 0.0 and 1.0
        """
        try:
            # Extract task information
            if isinstance(task, str):
                task_text = task.lower()
                task_type = self._infer_task_type(task_text)
                task_keywords = self._extract_keywords(task_text)
                urgency = 0.5  # Default urgency
                complexity = self._estimate_complexity(task_text)
            else:
                task_text = task.get('description', '').lower()
                task_type = task.get('type', self._infer_task_type(task_text))
                task_keywords = task.get('keywords', self._extract_keywords(task_text))
                urgency = task.get('urgency', 0.5)
                complexity = task.get('complexity', self._estimate_complexity(task_text))
            
            # Calculate alignment components
            capability_score = self._score_capability_match(task_keywords, task_type)
            preference_score = self._score_preference_match(task_keywords, task_type)
            confidence_score = self._score_confidence_match(task_type, complexity)
            condition_score = self._score_ideal_conditions(urgency, complexity)
            weakness_penalty = self._score_weakness_penalty(task_keywords, task_type)
            
            # Weighted combination
            weights = {
                'capability': 0.3,
                'preference': 0.2,
                'confidence': 0.25,
                'conditions': 0.15,
                'weakness_penalty': 0.1
            }
            
            raw_score = (
                weights['capability'] * capability_score +
                weights['preference'] * preference_score +
                weights['confidence'] * confidence_score +
                weights['conditions'] * condition_score -
                weights['weakness_penalty'] * weakness_penalty
            )
            
            # Ensure score is in [0, 1] range
            aligned_score = max(0.0, min(1.0, raw_score))
            
            # Log alignment details for debugging
            logging.debug(f"[{self.agent_name}Contract] Task alignment: {aligned_score:.3f} "
                         f"(cap:{capability_score:.2f}, pref:{preference_score:.2f}, "
                         f"conf:{confidence_score:.2f}, cond:{condition_score:.2f}, "
                         f"weak:{weakness_penalty:.2f})")
            
            return aligned_score
            
        except Exception as e:
            logging.error(f"[{self.agent_name}Contract] Error scoring alignment: {e}")
            return 0.0
    
    def _infer_task_type(self, task_text: str) -> str:
        """Infer the primary type of task from text."""
        task_patterns = {
            'planning': ['plan', 'organize', 'structure', 'breakdown', 'strategy', 'roadmap'],
            'analysis': ['analyze', 'examine', 'review', 'assess', 'evaluate', 'investigate'],
            'debugging': ['debug', 'fix', 'error', 'bug', 'issue', 'problem', 'troubleshoot'],
            'optimization': ['optimize', 'improve', 'enhance', 'refactor', 'performance'],
            'creative': ['create', 'generate', 'design', 'invent', 'brainstorm', 'ideate'],
            'execution': ['implement', 'execute', 'run', 'perform', 'carry out', 'deploy'],
            'communication': ['explain', 'communicate', 'present', 'clarify', 'discuss']
        }
        
        for task_type, keywords in task_patterns.items():
            if any(keyword in task_text for keyword in keywords):
                return task_type
        
        return 'general'
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract relevant keywords from task text."""
        # Simple keyword extraction - could be enhanced with NLP
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        keywords = {word for word in words if len(word) > 2 and word not in stop_words}
        return keywords
    
    def _estimate_complexity(self, task_text: str) -> float:
        """Estimate task complexity from 0.0 to 1.0."""
        complexity_indicators = {
            'simple': ['simple', 'easy', 'basic', 'quick', 'small'],
            'medium': ['moderate', 'standard', 'normal', 'regular'],
            'complex': ['complex', 'complicated', 'difficult', 'challenging', 'advanced', 'large', 'comprehensive']
        }
        
        for level, indicators in complexity_indicators.items():
            if any(indicator in task_text for indicator in indicators):
                if level == 'simple':
                    return 0.3
                elif level == 'medium':
                    return 0.6
                else:  # complex
                    return 0.9
        
        # Default complexity based on text length and technical terms
        technical_terms = ['algorithm', 'implementation', 'architecture', 'system', 'framework', 'integration']
        tech_score = sum(1 for term in technical_terms if term in task_text) / len(technical_terms)
        length_score = min(1.0, len(task_text.split()) / 50.0)
        
        return (tech_score + length_score) / 2
    
    def _score_capability_match(self, keywords: Set[str], task_type: str) -> float:
        """Score how well agent capabilities match the task."""
        capability_keywords = {cap.lower() for cap in self.capabilities}
        keyword_overlap = len(keywords.intersection(capability_keywords))
        
        # Type-specific capability matching
        type_match = 1.0 if task_type in [cap.lower() for cap in self.capabilities] else 0.0
        
        # Combine keyword and type matching
        keyword_score = min(1.0, keyword_overlap / max(1, len(keywords) * 0.3))
        
        return (type_match * 0.7) + (keyword_score * 0.3)
    
    def _score_preference_match(self, keywords: Set[str], task_type: str) -> float:
        """Score how well the task matches agent preferences."""
        if not self.preferred_tasks:
            return 0.5  # Neutral if no preferences specified
        
        preference_keywords = {pref.lower() for pref in self.preferred_tasks}
        keyword_overlap = len(keywords.intersection(preference_keywords))
        
        # Type preference matching
        type_match = 1.0 if task_type in [pref.lower() for pref in self.preferred_tasks] else 0.0
        
        keyword_score = min(1.0, keyword_overlap / max(1, len(keywords) * 0.4))
        
        return (type_match * 0.6) + (keyword_score * 0.4)
    
    def _score_confidence_match(self, task_type: str, complexity: float) -> float:
        """Score based on agent confidence for this type of task."""
        # Map task type to confidence vector keys
        confidence_mapping = {
            'planning': 'planning',
            'analysis': 'analysis',
            'debugging': 'debugging',
            'optimization': 'optimization',
            'creative': 'creativity',
            'execution': 'execution',
            'communication': 'communication',
            'general': 'problem_solving'
        }
        
        confidence_key = confidence_mapping.get(task_type, 'problem_solving')
        base_confidence = self.confidence_vector.get(confidence_key, 0.5)
        
        # Adjust confidence based on task complexity
        complexity_adjustment = 1.0 - (complexity * 0.3)  # Reduce confidence for complex tasks
        
        return base_confidence * complexity_adjustment
    
    def _score_ideal_conditions(self, urgency: float, complexity: float) -> float:
        """Score based on ideal operating conditions."""
        if not self.ideal_conditions:
            return 0.5
        
        score = 0.5
        
        # Check urgency preference
        max_urgency = self.ideal_conditions.get('max_urgency', 1.0)
        if urgency <= max_urgency:
            score += 0.2
        
        # Check complexity preference
        max_complexity = self.ideal_conditions.get('max_complexity', 1.0)
        if complexity <= max_complexity:
            score += 0.2
        
        # Check other conditions
        min_quality_time = self.ideal_conditions.get('min_quality_time', 0)
        if min_quality_time == 0 or urgency < 0.8:  # Assume high urgency means less quality time
            score += 0.1
        
        return min(1.0, score)
    
    def _score_weakness_penalty(self, keywords: Set[str], task_type: str) -> float:
        """Calculate penalty based on known weaknesses."""
        if not self.weaknesses:
            return 0.0
        
        weakness_keywords = {weak.lower() for weak in self.weaknesses}
        keyword_overlap = len(keywords.intersection(weakness_keywords))
        
        # Type weakness matching
        type_penalty = 1.0 if task_type in [weak.lower() for weak in self.weaknesses] else 0.0
        
        keyword_penalty = min(1.0, keyword_overlap / max(1, len(keywords) * 0.5))
        
        return (type_penalty * 0.7) + (keyword_penalty * 0.3)
    
    def update_from_performance_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Update contract based on performance feedback.
        
        Args:
            feedback: Dictionary containing performance metrics and outcomes
        """
        try:
            # Record performance history
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'task_type': feedback.get('task_type', 'unknown'),
                'success': feedback.get('success', False),
                'performance_score': feedback.get('performance_score', 0.5),
                'completion_time': feedback.get('completion_time', 0),
                'quality_rating': feedback.get('quality_rating', 0.5),
                'notes': feedback.get('notes', '')
            }
            
            self.performance_history.append(feedback_entry)
            
            # Limit history size
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
            # Update confidence vector based on recent performance
            self._update_confidence_from_feedback(feedback)
            
            # Track specialization drift
            self._track_specialization_drift(feedback)
            
            # Update timestamp
            self.last_updated = datetime.now()
            
            logging.info(f"[{self.agent_name}Contract] Updated from performance feedback: "
                        f"success={feedback.get('success')}, score={feedback.get('performance_score', 0.5):.2f}")
            
        except Exception as e:
            logging.error(f"[{self.agent_name}Contract] Error updating from feedback: {e}")
    
    def _update_confidence_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """Update confidence vector based on performance feedback."""
        task_type = feedback.get('task_type', 'general')
        success = feedback.get('success', False)
        performance_score = feedback.get('performance_score', 0.5)
        
        # Map task type to confidence key
        confidence_mapping = {
            'planning': 'planning',
            'analysis': 'analysis',
            'debugging': 'debugging',
            'optimization': 'optimization',
            'creative': 'creativity',
            'execution': 'execution',
            'communication': 'communication',
            'general': 'problem_solving'
        }
        
        confidence_key = confidence_mapping.get(task_type, 'problem_solving')
        
        if confidence_key in self.confidence_vector:
            current_confidence = self.confidence_vector[confidence_key]
            
            # Calculate adjustment based on performance
            if success and performance_score > 0.7:
                # Good performance - increase confidence slightly
                adjustment = 0.05 * performance_score
            elif success and performance_score > 0.5:
                # Moderate performance - small increase
                adjustment = 0.02 * performance_score
            elif not success:
                # Poor performance - decrease confidence
                adjustment = -0.03 * (1.0 - performance_score)
            else:
                adjustment = 0.0
            
            # Apply adjustment with bounds
            new_confidence = max(0.0, min(1.0, current_confidence + adjustment))
            self.confidence_vector[confidence_key] = new_confidence
            
            logging.debug(f"[{self.agent_name}Contract] Updated {confidence_key} confidence: "
                         f"{current_confidence:.3f} -> {new_confidence:.3f}")
    
    def _track_specialization_drift(self, feedback: Dict[str, Any]) -> None:
        """Track how agent specialization is drifting over time."""
        task_type = feedback.get('task_type', 'general')
        performance_score = feedback.get('performance_score', 0.5)
        
        if task_type not in self.specialization_drift:
            self.specialization_drift[task_type] = []
        
        self.specialization_drift[task_type].append({
            'timestamp': datetime.now().isoformat(),
            'performance': performance_score
        })
        
        # Keep only recent drift data (last 50 entries per type)
        if len(self.specialization_drift[task_type]) > 50:
            self.specialization_drift[task_type] = self.specialization_drift[task_type][-50:]
    
    def get_specialization_trend(self, task_type: str, window_size: int = 10) -> Optional[float]:
        """
        Get the recent performance trend for a specific task type.
        
        Args:
            task_type: Type of task to analyze
            window_size: Number of recent entries to analyze
            
        Returns:
            Trend score: positive = improving, negative = declining, None = insufficient data
        """
        if task_type not in self.specialization_drift:
            return None
        
        data = self.specialization_drift[task_type]
        if len(data) < 3:
            return None
        
        recent_data = data[-window_size:]
        if len(recent_data) < 3:
            recent_data = data
        
        # Calculate simple trend (difference between recent average and older average)
        mid_point = len(recent_data) // 2
        older_avg = sum(entry['performance'] for entry in recent_data[:mid_point]) / mid_point
        recent_avg = sum(entry['performance'] for entry in recent_data[mid_point:]) / (len(recent_data) - mid_point)
        
        return recent_avg - older_avg
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert contract to dictionary for serialization."""
        return {
            'agent_name': self.agent_name,
            'purpose': self.purpose,
            'capabilities': self.capabilities,
            'preferred_tasks': self.preferred_tasks,
            'weaknesses': self.weaknesses,
            'confidence_vector': self.confidence_vector,
            'version': self.version,
            'last_updated': self.last_updated.isoformat(),
            'performance_history': self.performance_history,
            'specialization_drift': self.specialization_drift,
            'ideal_conditions': self.ideal_conditions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentContract':
        """Create contract from dictionary."""
        # Parse datetime
        last_updated = datetime.fromisoformat(data.get('last_updated', datetime.now().isoformat()))
        
        return cls(
            agent_name=data['agent_name'],
            purpose=data['purpose'],
            capabilities=data.get('capabilities', []),
            preferred_tasks=data.get('preferred_tasks', []),
            weaknesses=data.get('weaknesses', []),
            confidence_vector=data.get('confidence_vector', {}),
            version=data.get('version', '1.0.0'),
            last_updated=last_updated,
            performance_history=data.get('performance_history', []),
            specialization_drift=data.get('specialization_drift', {}),
            ideal_conditions=data.get('ideal_conditions', {})
        )
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save contract to JSON file."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            
            logging.info(f"[{self.agent_name}Contract] Saved to {filepath}")
            
        except Exception as e:
            logging.error(f"[{self.agent_name}Contract] Error saving to {filepath}: {e}")
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> Optional['AgentContract']:
        """Load contract from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            contract = cls.from_dict(data)
            logging.info(f"[{contract.agent_name}Contract] Loaded from {filepath}")
            return contract
            
        except FileNotFoundError:
            logging.warning(f"[AgentContract] Contract file not found: {filepath}")
            return None
        except Exception as e:
            logging.error(f"[AgentContract] Error loading from {filepath}: {e}")
            return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the contract for display."""
        recent_performance = None
        if self.performance_history:
            recent_entries = self.performance_history[-10:]
            success_rate = sum(1 for entry in recent_entries if entry.get('success', False)) / len(recent_entries)
            avg_performance = sum(entry.get('performance_score', 0.5) for entry in recent_entries) / len(recent_entries)
            recent_performance = {
                'success_rate': success_rate,
                'avg_performance': avg_performance,
                'total_tasks': len(self.performance_history)
            }
        
        top_capabilities = sorted(self.confidence_vector.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'agent_name': self.agent_name,
            'purpose': self.purpose,
            'version': self.version,
            'last_updated': self.last_updated.isoformat(),
            'top_capabilities': top_capabilities,
            'total_capabilities': len(self.capabilities),
            'preferred_tasks': len(self.preferred_tasks),
            'known_weaknesses': len(self.weaknesses),
            'recent_performance': recent_performance
        }


def create_default_contracts() -> Dict[str, AgentContract]:
    """Create default contracts for all known agent types."""
    
    contracts = {}
    
    # Planner Agent Contract
    contracts['planner'] = AgentContract(
        agent_name='planner',
        purpose='Break down complex tasks into clear, actionable steps with logical sequencing and resource identification',
        capabilities=['task_decomposition', 'step_sequencing', 'resource_identification', 'timeline_estimation', 'dependency_analysis'],
        preferred_tasks=['planning', 'organization', 'project_breakdown', 'strategy_development', 'milestone_creation'],
        weaknesses=['execution_details', 'real_time_adaptation', 'context_switching'],
        confidence_vector={'planning': 0.9, 'reasoning': 0.8, 'execution': 0.6, 'analysis': 0.7, 'problem_solving': 0.8},
        ideal_conditions={'max_complexity': 0.8, 'max_urgency': 0.7, 'min_quality_time': 30}
    )
    
    # Critic Agent Contract
    contracts['critic'] = AgentContract(
        agent_name='critic',
        purpose='Identify flaws, weaknesses, and potential issues through thorough critical analysis',
        capabilities=['critical_analysis', 'risk_identification', 'assumption_challenging', 'flaw_detection', 'quality_assessment'],
        preferred_tasks=['analysis', 'review', 'evaluation', 'risk_assessment', 'quality_control'],
        weaknesses=['solution_generation', 'positive_reinforcement', 'quick_decisions'],
        confidence_vector={'analysis': 0.9, 'reasoning': 0.8, 'problem_solving': 0.7, 'communication': 0.6, 'creativity': 0.4},
        ideal_conditions={'max_complexity': 0.9, 'max_urgency': 0.5, 'min_quality_time': 45}
    )
    
    # Debater Agent Contract
    contracts['debater'] = AgentContract(
        agent_name='debater',
        purpose='Engage in structured debate and argument analysis to explore multiple perspectives',
        capabilities=['argument_construction', 'perspective_analysis', 'debate_facilitation', 'logic_evaluation', 'position_advocacy'],
        preferred_tasks=['debate', 'discussion', 'argument_analysis', 'perspective_exploration', 'conflict_resolution'],
        weaknesses=['consensus_building', 'emotional_intelligence', 'quick_agreement'],
        confidence_vector={'communication': 0.9, 'reasoning': 0.8, 'analysis': 0.7, 'creativity': 0.6, 'problem_solving': 0.7},
        ideal_conditions={'max_complexity': 0.7, 'max_urgency': 0.6, 'min_quality_time': 60}
    )
    
    # Reflector Agent Contract
    contracts['reflector'] = AgentContract(
        agent_name='reflector',
        purpose='Provide deep introspection and self-analysis for continuous improvement',
        capabilities=['introspection', 'belief_analysis', 'contradiction_resolution', 'self_assessment', 'pattern_recognition'],
        preferred_tasks=['reflection', 'self_analysis', 'belief_examination', 'contradiction_resolution', 'improvement_identification'],
        weaknesses=['external_focus', 'action_orientation', 'quick_decisions'],
        confidence_vector={'analysis': 0.8, 'reasoning': 0.9, 'problem_solving': 0.7, 'planning': 0.6, 'creativity': 0.7},
        ideal_conditions={'max_complexity': 0.8, 'max_urgency': 0.4, 'min_quality_time': 90}
    )
    
    # Add more default contracts for other agents
    additional_agents = {
        'architect_analyzer': {
            'purpose': 'Analyze and design system architecture with focus on scalability and maintainability',
            'capabilities': ['architecture_design', 'system_analysis', 'scalability_planning', 'pattern_recognition', 'design_optimization'],
            'preferred_tasks': ['architecture', 'design', 'system_planning', 'optimization', 'technical_analysis'],
            'weaknesses': ['implementation_details', 'user_interface', 'quick_prototyping'],
            'confidence_vector': {'planning': 0.8, 'domain_expertise': 0.9, 'optimization': 0.8, 'analysis': 0.8, 'reasoning': 0.7}
        },
        'code_refactorer': {
            'purpose': 'Improve code quality through refactoring, optimization, and best practices application',
            'capabilities': ['code_refactoring', 'optimization', 'best_practices', 'quality_improvement', 'technical_debt_reduction'],
            'preferred_tasks': ['refactoring', 'optimization', 'code_improvement', 'debugging', 'quality_enhancement'],
            'weaknesses': ['new_feature_development', 'user_requirements', 'business_logic'],
            'confidence_vector': {'debugging': 0.9, 'optimization': 0.8, 'execution': 0.8, 'analysis': 0.7, 'domain_expertise': 0.8}
        },
        'world_modeler': {
            'purpose': 'Model and understand complex systems and their interactions in the world',
            'capabilities': ['system_modeling', 'relationship_analysis', 'world_understanding', 'context_analysis', 'predictive_modeling'],
            'preferred_tasks': ['modeling', 'analysis', 'prediction', 'system_understanding', 'context_building'],
            'weaknesses': ['specific_implementation', 'detailed_execution', 'user_interaction'],
            'confidence_vector': {'domain_expertise': 0.9, 'reasoning': 0.8, 'analysis': 0.8, 'problem_solving': 0.7, 'planning': 0.6}
        },
        'constraint_engine': {
            'purpose': 'Manage and optimize complex constraints to find feasible solutions',
            'capabilities': ['constraint_satisfaction', 'optimization', 'feasibility_analysis', 'solution_space_exploration', 'trade_off_analysis'],
            'preferred_tasks': ['optimization', 'constraint_solving', 'feasibility_analysis', 'trade_off_evaluation', 'solution_finding'],
            'weaknesses': ['creative_solutions', 'user_experience', 'communication'],
            'confidence_vector': {'optimization': 0.9, 'reasoning': 0.8, 'problem_solving': 0.8, 'analysis': 0.7, 'domain_expertise': 0.7}
        },
        'self_healer': {
            'purpose': 'Diagnose and repair system issues through self-diagnosis and healing mechanisms',
            'capabilities': ['self_diagnosis', 'error_detection', 'automatic_repair', 'system_monitoring', 'recovery_strategies'],
            'preferred_tasks': ['debugging', 'error_handling', 'system_repair', 'monitoring', 'recovery'],
            'weaknesses': ['prevention', 'user_training', 'documentation'],
            'confidence_vector': {'debugging': 0.9, 'analysis': 0.8, 'problem_solving': 0.9, 'execution': 0.7, 'reasoning': 0.7}
        }
    }
    
    for agent_name, config in additional_agents.items():
        contracts[agent_name] = AgentContract(
            agent_name=agent_name,
            purpose=config['purpose'],
            capabilities=config['capabilities'],
            preferred_tasks=config['preferred_tasks'],
            weaknesses=config['weaknesses'],
            confidence_vector=config['confidence_vector'],
            ideal_conditions={'max_complexity': 0.8, 'max_urgency': 0.6, 'min_quality_time': 60}
        )
    
    return contracts


def load_or_create_contract(agent_name: str, contracts_dir: Union[str, Path] = "output/contracts") -> AgentContract:
    """
    Load an existing contract or create a default one for the given agent.
    
    Args:
        agent_name: Name of the agent
        contracts_dir: Directory containing contract files
        
    Returns:
        AgentContract instance
    """
    contracts_dir = Path(contracts_dir)
    contract_file = contracts_dir / f"{agent_name}.json"
    
    # Try to load existing contract
    contract = AgentContract.load_from_file(contract_file)
    
    if contract is None:
        # Create default contract
        default_contracts = create_default_contracts()
        if agent_name in default_contracts:
            contract = default_contracts[agent_name]
        else:
            # Create minimal contract for unknown agents
            contract = AgentContract(
                agent_name=agent_name,
                purpose=f"Specialized cognitive agent for {agent_name} tasks",
                capabilities=[agent_name, 'general_processing'],
                preferred_tasks=[agent_name, 'general_tasks'],
                weaknesses=['unknown_domains'],
                confidence_vector={'problem_solving': 0.5, 'reasoning': 0.5}
            )
        
        # Save the default contract
        contract.save_to_file(contract_file)
        logging.info(f"[AgentContract] Created default contract for {agent_name}")
    
    return contract


def score_all_agents_for_task(task: Union[str, Dict[str, Any]], 
                             contracts_dir: Union[str, Path] = "output/contracts") -> List[Dict[str, Any]]:
    """
    Score all available agents for a given task.
    
    Args:
        task: Task description or task object
        contracts_dir: Directory containing contract files
        
    Returns:
        List of dicts with agent names and their alignment scores, sorted by score
    """
    contracts_dir = Path(contracts_dir)
    agent_scores = []
    
    if not contracts_dir.exists():
        logging.warning(f"[AgentContract] Contracts directory not found: {contracts_dir}")
        return agent_scores
    
    # Load all contracts and score them
    for contract_file in contracts_dir.glob("*.json"):
        try:
            contract = AgentContract.load_from_file(contract_file)
            if contract:
                score = contract.score_alignment(task)
                agent_scores.append({
                    'agent_name': contract.agent_name,
                    'alignment_score': score,
                    'purpose': contract.purpose,
                    'confidence_vector': contract.confidence_vector
                })
        except Exception as e:
            logging.error(f"[AgentContract] Error scoring {contract_file}: {e}")
    
    # Sort by alignment score (highest first)
    agent_scores.sort(key=lambda x: x['alignment_score'], reverse=True)
    
    return agent_scores