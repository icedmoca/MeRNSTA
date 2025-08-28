#!/usr/bin/env python3
"""
SelfPromptGenerator - Autonomous goal generation agent for MeRNSTA

Analyzes memory, performance patterns, and reflection logs to autonomously
generate self-directed improvement goals. Enables true self-directed learning
and continuous improvement without external prompting.

Phase 13 Enhancement: Added autonomous command generation and execution from self-prompts.
"""

import logging
import json
import uuid
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter

from .base import BaseAgent

# Phase 13: Import command routing capabilities
try:
    from .command_router import route_command_async
    from config.settings import get_config
    COMMAND_ROUTER_AVAILABLE = True
except ImportError:
    COMMAND_ROUTER_AVAILABLE = False
    logging.warning("[SelfPromptGenerator] Command router not available - autonomous commands disabled")

class SelfPromptGenerator(BaseAgent):
    """
    Autonomous goal generation agent that analyzes patterns to create self-directed goals.
    
    Capabilities:
    - Memory analysis for improvement opportunities
    - Pattern recognition in failures and successes
    - Self-reflection goal generation
    - Priority-based goal ranking
    - Learning from past performance
    """
    
    def __init__(self):
        super().__init__("self_prompter")
        
        # Configuration from config.yaml patterns [[memory:4199483]]
        self.analysis_lookback_days = self.agent_config.get('analysis_lookback_days', 7)
        self.max_goals_per_session = self.agent_config.get('max_goals_per_session', 5)
        self.min_confidence_threshold = self.agent_config.get('min_confidence', 0.4)
        self.failure_weight = self.agent_config.get('failure_weight', 1.5)
        self.success_pattern_weight = self.agent_config.get('success_pattern_weight', 0.8)
        
        # Initialize connections to other systems
        self._memory_system = None
        self._plan_memory = None
        self._intention_model = None
        self._reflection_system = None
        self._self_healer = None
        
        # Phase 13: Command generation configuration
        self.command_config = get_config().get('autonomous_commands', {}) if COMMAND_ROUTER_AVAILABLE else {}
        self.enable_self_prompt_commands = self.command_config.get('enable_self_prompt_commands', False)
        self.max_commands_per_session = self.command_config.get('max_commands_per_cycle', 3)
        self.require_confirmation = self.command_config.get('require_confirmation', True)
        
        # Command generation patterns for self-improvement
        self.self_improvement_patterns = {
            'learning_deficit': r'(?i)(don\'t know|unclear|confused|need to learn)',
            'performance_issue': r'(?i)(slow|inefficient|poor performance|optimize)',
            'error_pattern': r'(?i)(error|bug|fail|broken|exception)',
            'knowledge_gap': r'(?i)(missing|incomplete|gap|understand better)',
            'skill_improvement': r'(?i)(practice|improve|enhance|develop|master)',
            'system_maintenance': r'(?i)(clean|organize|update|maintain|backup)',
        }
        
        # Goal generation patterns (dynamically loaded from config)
        self.goal_patterns = self._load_goal_patterns()
        
        logging.info(f"[{self.name}] Initialized with {len(self.goal_patterns)} goal patterns")
    
    def _load_goal_patterns(self) -> Dict[str, Any]:
        """Load goal generation patterns from configuration [[memory:4199483]]"""
        default_patterns = {
            'improvement_areas': [
                'memory_efficiency', 'response_accuracy', 'planning_quality',
                'reflection_depth', 'contradiction_resolution', 'learning_speed',
                'pattern_recognition', 'goal_achievement', 'self_awareness'
            ],
            'failure_analysis_patterns': [
                'repeated_mistakes', 'incomplete_tasks', 'low_confidence_areas',
                'contradiction_clusters', 'abandoned_goals', 'timeout_failures'
            ],
            'success_amplification_patterns': [
                'high_scoring_methods', 'efficient_approaches', 'successful_strategies',
                'effective_patterns', 'rapid_learning_areas', 'consistent_performance'
            ],
            'exploration_patterns': [
                'untested_approaches', 'knowledge_gaps', 'novel_combinations',
                'edge_case_handling', 'creative_solutions', 'experimental_methods'
            ],
            'maintenance_patterns': [
                'system_optimization', 'data_cleanup', 'performance_monitoring',
                'capability_assessment', 'resource_management', 'health_checks'
            ]
        }
        
        # Get patterns from config, fallback to defaults [[memory:4199483]]
        config_patterns = self.agent_config.get('goal_patterns', {})
        return {**default_patterns, **config_patterns}
    
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
                    logging.warning(f"[{self.name}] No memory system available")
        return self._memory_system
    
    @property
    def plan_memory(self):
        """Lazy-load plan memory"""
        if self._plan_memory is None:
            try:
                from storage.plan_memory import PlanMemory
                self._plan_memory = PlanMemory()
            except ImportError:
                logging.warning(f"[{self.name}] Plan memory not available")
        return self._plan_memory
    
    @property
    def intention_model(self):
        """Lazy-load intention model"""
        if self._intention_model is None:
            try:
                from storage.intention_model import IntentionModel
                self._intention_model = IntentionModel()
            except ImportError:
                logging.warning(f"[{self.name}] Intention model not available")
        return self._intention_model
    
    @property
    def reflection_system(self):
        """Lazy-load reflection system"""
        if self._reflection_system is None:
            try:
                from agents.reflective_engine import ReflectiveEngine
                self._reflection_system = ReflectiveEngine()
            except ImportError:
                logging.warning(f"[{self.name}] Reflection system not available")
        return self._reflection_system
    
    @property
    def self_healer(self):
        """Lazy-load self-healer system"""
        if self._self_healer is None:
            try:
                from agents.self_healer import SelfHealer
                self._self_healer = SelfHealer()
            except ImportError:
                logging.warning(f"[{self.name}] Self-healer not available")
        return self._self_healer
    
    def get_agent_instructions(self) -> str:
        """Return specialized instructions for the self-prompt generator."""
        return (
            "You are an autonomous goal generation specialist. Your role is to analyze "
            "memory patterns, performance data, and reflection logs to identify opportunities "
            "for self-improvement. Generate meaningful, actionable goals that will enhance "
            "the system's capabilities without external prompting. Focus on learning from "
            "failures, amplifying successes, and exploring new possibilities. "
            f"Analysis lookback: {self.analysis_lookback_days} days. "
            f"Max goals per session: {self.max_goals_per_session}."
        )
    
    def propose_goals(self, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Analyze system state and propose self-directed improvement goals.
        
        Args:
            context: Additional context for goal generation
            
        Returns:
            List of proposed goal descriptions
        """
        if not self.enabled:
            return []
        
        try:
            # Gather analysis data
            analysis_data = self._gather_analysis_data(context)
            
            # Generate goals from different sources
            proposed_goals = []
            
            # Self-repair goals (highest priority)
            self_repair_goals = self._generate_self_repair_goals(analysis_data)
            proposed_goals.extend(self_repair_goals)
            
            # Memory-based goals
            memory_goals = self._generate_memory_based_goals(analysis_data)
            proposed_goals.extend(memory_goals)
            
            # Performance-based goals
            performance_goals = self._generate_performance_based_goals(analysis_data)
            proposed_goals.extend(performance_goals)
            
            # Pattern-based goals
            pattern_goals = self._generate_pattern_based_goals(analysis_data)
            proposed_goals.extend(pattern_goals)
            
            # Exploration goals
            exploration_goals = self._generate_exploration_goals(analysis_data)
            proposed_goals.extend(exploration_goals)
            
            # Maintenance goals
            maintenance_goals = self._generate_maintenance_goals(analysis_data)
            proposed_goals.extend(maintenance_goals)
            
            # Filter and prioritize
            filtered_goals = self._filter_and_deduplicate_goals(proposed_goals)
            prioritized_goals = self.prioritize_goals(filtered_goals)
            
            # Limit to max goals per session
            final_goals = prioritized_goals[:self.max_goals_per_session]
            
            logging.info(f"[{self.name}] Generated {len(final_goals)} self-directed goals")
            
            return final_goals
            
        except Exception as e:
            logging.error(f"[{self.name}] Error proposing goals: {e}")
            return []
    
    def prioritize_goals(self, goals: List[str]) -> List[str]:
        """
        Prioritize goals based on impact, feasibility, and urgency.
        
        Args:
            goals: List of goal descriptions to prioritize
            
        Returns:
            Goals sorted by priority (highest first)
        """
        try:
            if not goals:
                return []
            
            # Score each goal
            goal_scores = []
            for goal in goals:
                score = self._score_goal_priority(goal)
                goal_scores.append((goal, score))
            
            # Sort by score (highest first)
            goal_scores.sort(key=lambda x: x[1], reverse=True)
            
            prioritized = [goal for goal, score in goal_scores]
            
            logging.info(f"[{self.name}] Prioritized {len(goals)} goals")
            
            return prioritized
            
        except Exception as e:
            logging.error(f"[{self.name}] Error prioritizing goals: {e}")
            return goals  # Return original order on error
    
    def _gather_analysis_data(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Gather data for goal generation analysis"""
        cutoff_date = datetime.now() - timedelta(days=self.analysis_lookback_days)
        
        analysis_data = {
            'context': context or {},
            'cutoff_date': cutoff_date.isoformat(),
            'memory_stats': {},
            'plan_stats': {},
            'intention_stats': {},
            'reflection_data': {},
            'performance_patterns': {},
            'system_health': {}
        }
        
        # Memory system analysis
        if self.memory_system:
            try:
                if hasattr(self.memory_system, 'get_contradiction_summary'):
                    analysis_data['memory_stats'] = {
                        'contradictions': self.memory_system.get_contradiction_summary(),
                        'recent_facts_count': self._get_recent_facts_count(),
                        'volatility_trends': self._get_volatility_trends()
                    }
            except Exception as e:
                logging.warning(f"[{self.name}] Could not gather memory stats: {e}")
        
        # Plan memory analysis
        if self.plan_memory:
            try:
                analysis_data['plan_stats'] = self.plan_memory.get_plan_statistics()
                analysis_data['failed_plans'] = self._get_recent_failed_plans()
                analysis_data['successful_patterns'] = self._get_successful_plan_patterns()
            except Exception as e:
                logging.warning(f"[{self.name}] Could not gather plan stats: {e}")
        
        # Intention model analysis
        if self.intention_model:
            try:
                analysis_data['intention_stats'] = self.intention_model.get_intention_statistics()
                analysis_data['motivational_patterns'] = self.intention_model.get_motivational_patterns()
            except Exception as e:
                logging.warning(f"[{self.name}] Could not gather intention stats: {e}")
        
        # Reflection system analysis
        if self.reflection_system and hasattr(self.reflection_system, 'get_recent_insights'):
            try:
                analysis_data['reflection_data'] = {
                    'recent_insights': self.reflection_system.get_recent_insights(days=self.analysis_lookback_days),
                    'pattern_discoveries': self._get_pattern_discoveries()
                }
            except Exception as e:
                logging.warning(f"[{self.name}] Could not gather reflection data: {e}")
        
        # System health analysis (for self-repair goals)
        if self.self_healer:
            try:
                # Get basic health metrics without full diagnostic (for performance)
                health_issues = self.self_healer.analyze_code_health()
                health_patterns = self.self_healer.detect_architecture_flaws()
                
                analysis_data['system_health'] = {
                    'total_issues': len(health_issues),
                    'critical_issues': len([i for i in health_issues if i.severity == 'critical']),
                    'high_issues': len([i for i in health_issues if i.severity == 'high']),
                    'patterns_detected': len(health_patterns),
                    'critical_patterns': len([p for p in health_patterns if p.risk_level == 'critical']),
                    'issues_by_category': {category: len([i for i in health_issues if i.category == category]) 
                                         for category in ['code_quality', 'architecture', 'performance', 'reliability', 'security']},
                    'recent_diagnostic_available': self.self_healer._last_diagnostic is not None
                }
            except Exception as e:
                logging.warning(f"[{self.name}] Could not gather system health data: {e}")
        
        return analysis_data
    
    def _generate_self_repair_goals(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate self-repair goals based on system health analysis"""
        goals = []
        
        try:
            if not self.self_healer:
                return goals
            
            # Run system diagnostic to identify issues
            diagnostic_report = self.self_healer.run_diagnostic_suite()
            
            # High priority issues get immediate repair goals
            critical_issues = [i for i in diagnostic_report.issues if i.severity == 'critical']
            high_issues = [i for i in diagnostic_report.issues if i.severity == 'high']
            
            if critical_issues:
                for issue in critical_issues[:3]:  # Top 3 critical issues
                    goals.append(f"URGENT: Fix critical {issue.category} issue in {issue.component}")
            
            # System health goals
            if diagnostic_report.system_health_score < 0.5:
                goals.append(f"Comprehensive system health improvement (current score: {diagnostic_report.system_health_score:.1%})")
            elif diagnostic_report.system_health_score < 0.7:
                goals.append(f"System health optimization to improve from {diagnostic_report.system_health_score:.1%}")
            
            # Architecture pattern goals
            critical_patterns = [p for p in diagnostic_report.patterns if p.risk_level == 'critical']
            if critical_patterns:
                for pattern in critical_patterns[:2]:  # Top 2 critical patterns
                    goals.append(f"Eliminate {pattern.name} anti-pattern ({pattern.frequency} occurrences)")
            
            # Component-specific goals for heavily affected areas
            if diagnostic_report.metrics.get('most_problematic_components'):
                top_component, issue_count = diagnostic_report.metrics['most_problematic_components'][0]
                if issue_count >= 5:
                    goals.append(f"Comprehensive refactoring of {top_component} ({issue_count} issues)")
            
            # Use generated repair goals from the diagnostic report
            if diagnostic_report.repair_goals:
                # Take top repair goals and mark them as self-generated
                for goal in diagnostic_report.repair_goals[:3]:
                    goals.append(f"Self-repair: {goal}")
            
            # Failed repair learning goals
            if hasattr(self.self_healer, 'repair_log') and self.self_healer.repair_log:
                try:
                    failed_repairs = self.self_healer.repair_log.get_failed_repairs(limit=5, days_back=7)
                    if len(failed_repairs) >= 3:
                        goals.append(f"Investigate and improve repair approach for {len(failed_repairs)} recent failed repairs")
                except Exception:
                    pass
            
            # Mark these as high priority by prefixing
            priority_goals = []
            for goal in goals:
                if not goal.startswith(('URGENT:', 'Self-repair:')):
                    priority_goals.append(f"HIGH PRIORITY: {goal}")
                else:
                    priority_goals.append(goal)
            
            logging.info(f"[{self.name}] Generated {len(priority_goals)} self-repair goals")
            
            return priority_goals
            
        except Exception as e:
            logging.error(f"[{self.name}] Error generating self-repair goals: {e}")
            return []
    
    def _generate_memory_based_goals(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate goals based on memory system analysis"""
        goals = []
        memory_stats = analysis_data.get('memory_stats', {})
        
        # Contradiction resolution goals
        contradictions = memory_stats.get('contradictions', {})
        if isinstance(contradictions, dict) and contradictions.get('count', 0) > 5:
            goals.append(f"Resolve {contradictions['count']} pending contradictions to improve knowledge consistency")
        
        # Memory optimization goals
        recent_facts = memory_stats.get('recent_facts_count', 0)
        if recent_facts > 100:
            goals.append("Implement memory compression strategy to optimize storage efficiency")
        
        # Volatility management goals
        volatility_trends = memory_stats.get('volatility_trends', {})
        if volatility_trends.get('high_volatility_count', 0) > 10:
            goals.append("Develop volatility prediction model to anticipate knowledge instability")
        
        return goals
    
    def _generate_performance_based_goals(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate goals based on performance analysis"""
        goals = []
        plan_stats = analysis_data.get('plan_stats', {})
        
        # Low success rate improvement
        avg_success_rate = plan_stats.get('average_success_rate', 1.0)
        if avg_success_rate < 0.7:
            goals.append(f"Improve plan execution success rate from {avg_success_rate:.1%} to >80%")
        
        # Failed plans analysis
        failed_plans = analysis_data.get('failed_plans', [])
        if len(failed_plans) > 3:
            common_failures = self._analyze_failure_patterns(failed_plans)
            if common_failures:
                goals.append(f"Address common failure pattern: {common_failures[0]}")
        
        # Intention fulfillment
        intention_stats = analysis_data.get('intention_stats', {})
        fulfillment_rate = intention_stats.get('fulfillment_rate', 1.0)
        if fulfillment_rate < 0.6:
            goals.append(f"Increase intention fulfillment rate from {fulfillment_rate:.1%} to >70%")
        
        return goals
    
    def _generate_pattern_based_goals(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate goals based on discovered patterns"""
        goals = []
        
        # Successful pattern amplification
        successful_patterns = analysis_data.get('successful_patterns', [])
        if successful_patterns:
            pattern = random.choice(successful_patterns)  # [[memory:4199483]] - dynamic selection
            goals.append(f"Scale successful pattern: {pattern['description']}")
        
        # Motivational pattern optimization
        motivational_patterns = analysis_data.get('motivational_patterns', {})
        drive_stats = motivational_patterns.get('drive_statistics', [])
        if drive_stats:
            low_success_drives = [d for d in drive_stats if d.get('success_rate', 1.0) < 0.5]
            if low_success_drives:
                drive = low_success_drives[0]['drive']
                goals.append(f"Improve success rate for {drive}-driven goals")
        
        return goals
    
    def _generate_exploration_goals(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate exploratory and experimental goals"""
        goals = []
        
        # Capability gaps exploration
        exploration_areas = self.goal_patterns.get('exploration_patterns', [])
        if exploration_areas:
            # Dynamically select unexplored areas [[memory:4199483]]
            area = random.choice(exploration_areas)
            goals.append(f"Explore {area} to expand current capabilities")
        
        # Novel approach testing
        reflection_data = analysis_data.get('reflection_data', {})
        insights = reflection_data.get('recent_insights', [])
        if insights:
            insight = random.choice(insights)
            goals.append(f"Test hypothesis from insight: {insight.get('summary', 'novel approach')}")
        
        # Cross-domain learning
        intention_stats = analysis_data.get('intention_stats', {})
        drive_breakdown = intention_stats.get('drive_breakdown', {})
        if len(drive_breakdown) > 2:
            # Find least used drive for exploration
            least_used = min(drive_breakdown.items(), key=lambda x: x[1])
            goals.append(f"Develop expertise in underutilized area: {least_used[0]}")
        
        return goals
    
    def _generate_maintenance_goals(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate system maintenance and optimization goals"""
        goals = []
        
        # Performance optimization
        maintenance_areas = self.goal_patterns.get('maintenance_patterns', [])
        if maintenance_areas:
            # Select based on current system state [[memory:4199483]]
            area = random.choice(maintenance_areas)
            goals.append(f"Conduct {area} to maintain optimal performance")
        
        # Data health checks
        memory_stats = analysis_data.get('memory_stats', {})
        if memory_stats:
            goals.append("Perform comprehensive memory system health audit")
        
        # Capability assessment
        plan_stats = analysis_data.get('plan_stats', {})
        executed_plans = plan_stats.get('executed_plans', 0)
        if executed_plans > 10:
            goals.append("Assess and document current planning capabilities")
        
        return goals
    
    def _filter_and_deduplicate_goals(self, goals: List[str]) -> List[str]:
        """Filter out invalid goals and remove duplicates"""
        if not goals:
            return []
        
        # Remove empty or too short goals
        filtered = [g for g in goals if g and len(g.strip()) > 10]
        
        # Remove near-duplicates using simple similarity
        deduplicated = []
        for goal in filtered:
            is_duplicate = False
            for existing in deduplicated:
                if self._goals_are_similar(goal, existing):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(goal)
        
        return deduplicated
    
    def _score_goal_priority(self, goal: str) -> float:
        """Score a goal for priority ranking"""
        score = 0.5  # Base score
        
        goal_lower = goal.lower()
        
        # Self-repair and system health goals get highest priority
        if any(prefix in goal_lower for prefix in ['urgent:', 'self-repair:', 'high priority:']):
            score += 0.4
        
        if any(keyword in goal_lower for keyword in ['system health', 'critical issue', 'architecture']):
            score += 0.3
        
        # High priority keywords
        high_priority_keywords = [
            'resolve', 'fix', 'critical', 'improve', 'optimize',
            'contradiction', 'failure', 'error', 'bottleneck', 'repair'
        ]
        for keyword in high_priority_keywords:
            if keyword in goal_lower:
                score += 0.1
        
        # Medium priority keywords
        medium_priority_keywords = [
            'develop', 'enhance', 'expand', 'learn', 'explore',
            'assess', 'analyze', 'understand'
        ]
        for keyword in medium_priority_keywords:
            if keyword in goal_lower:
                score += 0.05
        
        # Urgency indicators
        urgency_keywords = ['urgent', 'immediate', 'critical', 'failing']
        for keyword in urgency_keywords:
            if keyword in goal_lower:
                score += 0.2
        
        # Quantifiable goals get bonus
        if any(char.isdigit() for char in goal) or '%' in goal:
            score += 0.1
        
        # Specific improvement goals get bonus
        if 'rate' in goal_lower or 'performance' in goal_lower:
            score += 0.1
        
        return min(1.0, score)
    
    def _get_recent_facts_count(self) -> int:
        """Get count of recent facts in memory"""
        try:
            if hasattr(self.memory_system, 'get_facts_count_since'):
                cutoff = datetime.now() - timedelta(days=self.analysis_lookback_days)
                return self.memory_system.get_facts_count_since(cutoff)
            return 0
        except Exception:
            return 0
    
    def _get_volatility_trends(self) -> Dict[str, Any]:
        """Get volatility trends from memory"""
        try:
            if hasattr(self.memory_system, 'get_volatility_summary'):
                return self.memory_system.get_volatility_summary()
            return {}
        except Exception:
            return {}
    
    def _get_recent_failed_plans(self) -> List[Dict[str, Any]]:
        """Get recently failed plans"""
        try:
            if self.plan_memory:
                return self.plan_memory.get_plans_by_status('failed', limit=10)
            return []
        except Exception:
            return []
    
    def _get_successful_plan_patterns(self) -> List[Dict[str, Any]]:
        """Get patterns from successful plans"""
        try:
            if self.plan_memory:
                successful_plans = self.plan_memory.get_plans_by_status('completed', limit=20)
                # Analyze for common patterns
                patterns = []
                plan_types = Counter(p.get('plan_type', 'unknown') for p in successful_plans)
                for plan_type, count in plan_types.most_common(3):
                    patterns.append({
                        'type': 'plan_type',
                        'description': f"using {plan_type} planning approach",
                        'frequency': count
                    })
                return patterns
            return []
        except Exception:
            return []
    
    def _get_pattern_discoveries(self) -> List[Dict[str, Any]]:
        """Get recent pattern discoveries"""
        try:
            # This would integrate with pattern discovery systems
            # For now, return empty - implement when pattern discovery is available
            return []
        except Exception:
            return []
    
    def _analyze_failure_patterns(self, failed_plans: List[Dict[str, Any]]) -> List[str]:
        """Analyze common patterns in failed plans"""
        if not failed_plans:
            return []
        
        # Simple pattern analysis - could be enhanced with ML
        failure_reasons = []
        for plan in failed_plans:
            # Extract failure reasons from plan data
            # This would be enhanced with actual failure analysis
            plan_type = plan.get('plan_type', 'unknown')
            if plan_type != 'unknown':
                failure_reasons.append(f"{plan_type} planning approach")
        
        # Return most common patterns
        if failure_reasons:
            reason_counts = Counter(failure_reasons)
            return [reason for reason, count in reason_counts.most_common(3)]
        
        return []
    
    def _goals_are_similar(self, goal1: str, goal2: str, threshold: float = 0.7) -> bool:
        """Check if two goals are similar enough to be considered duplicates"""
        # Simple word overlap similarity
        words1 = set(goal1.lower().split())
        words2 = set(goal2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0.0
        return similarity >= threshold
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """Handle direct messages to the self-prompt generator"""
        if not self.enabled:
            return f"[{self.name}] Agent disabled"
        
        try:
            message_lower = message.lower()
            
            if any(phrase in message_lower for phrase in ['propose goals', 'suggest goals', 'generate goals']):
                goals = self.propose_goals(context)
                if goals:
                    response = f"🎯 **Self-Directed Goals Generated:**\n\n"
                    for i, goal in enumerate(goals, 1):
                        response += f"{i}. {goal}\n"
                    return response
                else:
                    return "No self-directed goals identified at this time."
            
            elif 'prioritize' in message_lower:
                # Extract goals from message or context
                goals = context.get('goals', []) if context else []
                if goals:
                    prioritized = self.prioritize_goals(goals)
                    response = "📊 **Prioritized Goals:**\n\n"
                    for i, goal in enumerate(prioritized, 1):
                        response += f"{i}. {goal}\n"
                    return response
                else:
                    return "No goals provided for prioritization."
            
            else:
                return self._provide_self_prompt_guidance(message, context)
        
        except Exception as e:
            logging.error(f"[{self.name}] Error in respond: {e}")
            return f"[{self.name}] Error: {str(e)}"
    
    # === PHASE 13: AUTONOMOUS COMMAND GENERATION ===
    
    async def generate_self_improvement_commands(self, goals: List[str], analysis_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Generate actionable commands from self-generated goals.
        
        Args:
            goals: List of self-improvement goals
            analysis_data: Analysis data from goal generation
            
        Returns:
            List of command dictionaries to execute
        """
        if not self.enable_self_prompt_commands or not COMMAND_ROUTER_AVAILABLE:
            return []
        
        commands = []
        
        try:
            for goal in goals:
                goal_commands = await self._analyze_goal_for_commands(goal, analysis_data)
                commands.extend(goal_commands)
            
            # Limit commands per session
            commands = commands[:self.max_commands_per_session]
            
            logging.info(f"[{self.name}] Generated {len(commands)} commands from {len(goals)} goals")
            return commands
            
        except Exception as e:
            logging.error(f"[{self.name}] Error generating self-improvement commands: {e}")
            return []
    
    async def _analyze_goal_for_commands(self, goal: str, analysis_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Analyze a goal and generate specific commands."""
        commands = []
        goal_lower = goal.lower()
        
        try:
            # Check for learning deficits
            if any(pattern in goal_lower for pattern in ['learn', 'understand', 'knowledge']):
                commands.extend(self._generate_learning_commands(goal))
            
            # Check for performance issues
            if any(pattern in goal_lower for pattern in ['performance', 'optimize', 'speed', 'efficient']):
                commands.extend(self._generate_performance_commands(goal))
            
            # Check for error patterns
            if any(pattern in goal_lower for pattern in ['error', 'bug', 'fix', 'repair']):
                commands.extend(self._generate_error_fix_commands(goal))
            
            # Check for knowledge gaps
            if any(pattern in goal_lower for pattern in ['gap', 'missing', 'incomplete']):
                commands.extend(self._generate_knowledge_gap_commands(goal))
            
            # Check for skill improvement
            if any(pattern in goal_lower for pattern in ['skill', 'practice', 'improve', 'develop']):
                commands.extend(self._generate_skill_commands(goal))
            
            # Check for system maintenance
            if any(pattern in goal_lower for pattern in ['maintain', 'clean', 'organize', 'update']):
                commands.extend(self._generate_maintenance_commands(goal))
            
        except Exception as e:
            logging.error(f"[{self.name}] Error analyzing goal for commands: {e}")
        
        return commands
    
    def _generate_learning_commands(self, goal: str) -> List[Dict[str, Any]]:
        """Generate learning-related commands."""
        commands = []
        
        if 'documentation' in goal.lower():
            commands.append({
                'command': '/run_tool read_file README.md',
                'purpose': f'Read documentation for learning goal: {goal}',
                'priority': 'medium',
                'goal': goal,
                'category': 'learning'
            })
        
        if 'python' in goal.lower():
            commands.append({
                'command': '/pip_install python-tutorial',
                'purpose': f'Install Python learning resources for goal: {goal}',
                'priority': 'medium',
                'goal': goal,
                'category': 'learning'
            })
        
        return commands
    
    def _generate_performance_commands(self, goal: str) -> List[Dict[str, Any]]:
        """Generate performance optimization commands."""
        commands = []
        
        commands.append({
            'command': '/run_shell "ps aux | sort -nk 3,3 | tail -10"',
            'purpose': f'Check system performance for goal: {goal}',
            'priority': 'high',
            'goal': goal,
            'category': 'performance'
        })
        
        if 'memory' in goal.lower():
            commands.append({
                'command': '/run_shell "free -h && df -h"',
                'purpose': f'Check memory and disk usage for goal: {goal}',
                'priority': 'medium',
                'goal': goal,
                'category': 'performance'
            })
        
        return commands
    
    def _generate_error_fix_commands(self, goal: str) -> List[Dict[str, Any]]:
        """Generate error fixing commands."""
        commands = []
        
        commands.append({
            'command': '/run_shell "grep -r "ERROR" logs/ | tail -20"',
            'purpose': f'Check recent errors for goal: {goal}',
            'priority': 'high',
            'goal': goal,
            'category': 'error_fix'
        })
        
        if 'test' in goal.lower():
            commands.append({
                'command': '/run_shell "python -m pytest tests/ -v"',
                'purpose': f'Run tests to identify issues for goal: {goal}',
                'priority': 'high',
                'goal': goal,
                'category': 'error_fix'
            })
        
        return commands
    
    def _generate_knowledge_gap_commands(self, goal: str) -> List[Dict[str, Any]]:
        """Generate commands to address knowledge gaps."""
        commands = []
        
        commands.append({
            'command': '/run_tool list_directory docs/',
            'purpose': f'Explore documentation to address knowledge gap: {goal}',
            'priority': 'medium',
            'goal': goal,
            'category': 'knowledge'
        })
        
        if 'config' in goal.lower():
            commands.append({
                'command': '/run_tool read_file config.yaml',
                'purpose': f'Review configuration for knowledge gap: {goal}',
                'priority': 'medium',
                'goal': goal,
                'category': 'knowledge'
            })
        
        return commands
    
    def _generate_skill_commands(self, goal: str) -> List[Dict[str, Any]]:
        """Generate skill development commands."""
        commands = []
        
        if 'coding' in goal.lower() or 'programming' in goal.lower():
            commands.append({
                'command': '/run_shell "find . -name "*.py" | xargs wc -l | sort -n | tail -10"',
                'purpose': f'Analyze codebase for skill development: {goal}',
                'priority': 'medium',
                'goal': goal,
                'category': 'skill'
            })
        
        return commands
    
    def _generate_maintenance_commands(self, goal: str) -> List[Dict[str, Any]]:
        """Generate system maintenance commands."""
        commands = []
        
        if 'clean' in goal.lower():
            commands.append({
                'command': '/run_shell "find . -name "*.pyc" -delete"',
                'purpose': f'Clean compiled files for maintenance goal: {goal}',
                'priority': 'low',
                'goal': goal,
                'category': 'maintenance'
            })
        
        if 'backup' in goal.lower():
            commands.append({
                'command': '/run_shell "tar -czf backup_$(date +%Y%m%d).tar.gz *.py *.yaml"',
                'purpose': f'Create backup for maintenance goal: {goal}',
                'priority': 'medium',
                'goal': goal,
                'category': 'maintenance'
            })
        
        if 'update' in goal.lower():
            commands.append({
                'command': '/pip_install --upgrade pip',
                'purpose': f'Update package manager for goal: {goal}',
                'priority': 'medium',
                'goal': goal,
                'category': 'maintenance'
            })
        
        return commands
    
    async def execute_self_improvement_commands(self, commands: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute commands generated from self-improvement goals.
        
        Args:
            commands: List of command dictionaries to execute
            
        Returns:
            Execution results and summary
        """
        if not COMMAND_ROUTER_AVAILABLE:
            return {'error': 'Command router not available'}
        
        execution_results = {
            'commands_executed': 0,
            'commands_successful': 0,
            'commands_failed': 0,
            'results_by_category': defaultdict(list),
            'results': [],
            'summary': ''
        }
        
        try:
            for cmd_info in commands:
                command = cmd_info['command']
                purpose = cmd_info.get('purpose', 'No purpose specified')
                priority = cmd_info.get('priority', 'medium')
                category = cmd_info.get('category', 'general')
                goal = cmd_info.get('goal', '')
                
                logging.info(f"[{self.name}] Executing self-improvement command: {command}")
                
                # Execute command
                result = await route_command_async(command, "self_prompter")
                
                execution_results['commands_executed'] += 1
                
                if result.get('success', False):
                    execution_results['commands_successful'] += 1
                    logging.info(f"[{self.name}] Command succeeded: {command}")
                else:
                    execution_results['commands_failed'] += 1
                    logging.warning(f"[{self.name}] Command failed: {command} - {result.get('error', 'Unknown error')}")
                
                cmd_result = {
                    'command': command,
                    'purpose': purpose,
                    'priority': priority,
                    'category': category,
                    'goal': goal,
                    'success': result.get('success', False),
                    'output': result.get('output', ''),
                    'error': result.get('error', '')
                }
                
                execution_results['results'].append(cmd_result)
                execution_results['results_by_category'][category].append(cmd_result)
            
            # Generate summary
            success_rate = execution_results['commands_successful'] / max(execution_results['commands_executed'], 1)
            execution_results['summary'] = (
                f"Executed {execution_results['commands_executed']} self-improvement commands. "
                f"Success rate: {success_rate:.1%} "
                f"({execution_results['commands_successful']} successful, {execution_results['commands_failed']} failed)"
            )
            
            logging.info(f"[{self.name}] {execution_results['summary']}")
            return execution_results
            
        except Exception as e:
            logging.error(f"[{self.name}] Error executing self-improvement commands: {e}")
            execution_results['summary'] = f"Command execution error: {str(e)}"
            return execution_results
    
    async def autonomous_self_improvement_cycle(self) -> Dict[str, Any]:
        """
        Run a complete autonomous self-improvement cycle.
        
        Returns:
            Results of the improvement cycle
        """
        try:
            logging.info(f"[{self.name}] Starting autonomous self-improvement cycle")
            
            # Generate goals
            goals = self.propose_goals()
            if not goals:
                return {'message': 'No self-improvement goals identified'}
            
            # Generate commands from goals
            commands = await self.generate_self_improvement_commands(goals)
            if not commands:
                return {'message': 'No actionable commands generated from goals'}
            
            # Execute commands if not requiring confirmation
            if not self.require_confirmation:
                execution_results = await self.execute_self_improvement_commands(commands)
                return {
                    'goals_generated': len(goals),
                    'commands_generated': len(commands),
                    'execution_results': execution_results,
                    'cycle_completed': True
                }
            else:
                return {
                    'goals_generated': len(goals),
                    'goals': goals,
                    'commands_generated': len(commands),
                    'commands': commands,
                    'awaiting_confirmation': True
                }
                
        except Exception as e:
            logging.error(f"[{self.name}] Error in autonomous self-improvement cycle: {e}")
            return {'error': str(e)}
    
    def _provide_self_prompt_guidance(self, message: str, context: Optional[Dict]) -> str:
        """Provide guidance on self-prompting capabilities"""
        guidance = [
            f"I'm the {self.name} agent. I can help with:",
            "• Analyzing system performance to identify improvement opportunities",
            "• Generating self-directed goals based on memory and performance patterns",
            "• Prioritizing goals by impact, feasibility, and urgency",
            "• Learning from failures to propose corrective actions",
            "• Discovering patterns in successful approaches for amplification",
            "",
            "Try: 'propose goals' or 'generate self-directed goals' to get started"
        ]
        
        return "\n".join(guidance)
    
    # === Phase 16: Planning Integration ===
    
    def generate_planning_improvement_goals(self) -> List[Dict[str, Any]]:
        """
        Generate goals specifically focused on improving planning capabilities.
        
        Returns:
            List of planning improvement goals with priorities and rationales
        """
        logging.info(f"[{self.name}] Generating planning improvement goals")
        
        planning_goals = []
        
        try:
            # Analyze current planning performance
            planning_analysis = self._analyze_planning_performance()
            
            # Generate goals based on performance gaps
            performance_goals = self._generate_goals_from_performance_gaps(planning_analysis)
            planning_goals.extend(performance_goals)
            
            # Generate goals for optimization opportunities
            optimization_goals = self._generate_optimization_goals(planning_analysis)
            planning_goals.extend(optimization_goals)
            
            # Generate goals for capability expansion
            expansion_goals = self._generate_capability_expansion_goals()
            planning_goals.extend(expansion_goals)
            
            # Score and prioritize all planning goals
            for goal in planning_goals:
                goal['score'] = self._score_planning_goal(goal)
            
            # Sort by score and limit to top goals
            planning_goals.sort(key=lambda g: g['score'], reverse=True)
            planning_goals = planning_goals[:self.max_goals_per_session]
            
            logging.info(f"[{self.name}] Generated {len(planning_goals)} planning improvement goals")
            
        except Exception as e:
            logging.error(f"[{self.name}] Error generating planning improvement goals: {e}")
            planning_goals.append({
                'goal_text': 'Improve planning goal generation error handling',
                'category': 'system_improvement',
                'priority': 'high',
                'rationale': f'Fix error in planning goal generation: {e}',
                'score': 0.8
            })
        
        return planning_goals
    
    def suggest_planning_optimizations(self, strategy_evaluation: Dict[str, Any]) -> List[str]:
        """
        Suggest specific optimizations based on strategy evaluation results.
        
        Args:
            strategy_evaluation: Results from strategy evaluation
            
        Returns:
            List of specific optimization suggestions
        """
        suggestions = []
        
        try:
            # Analyze evaluation results
            if 'criteria_scores' in strategy_evaluation:
                scores = strategy_evaluation['criteria_scores']
                
                # Suggest improvements for low-scoring criteria
                for criterion, score in scores.items():
                    if score < 0.6:
                        suggestion = self._generate_criterion_improvement_suggestion(criterion, score)
                        if suggestion:
                            suggestions.append(suggestion)
            
            # Analyze weaknesses
            if 'weaknesses' in strategy_evaluation:
                for weakness in strategy_evaluation['weaknesses']:
                    suggestion = self._generate_weakness_improvement_suggestion(weakness)
                    if suggestion:
                        suggestions.append(suggestion)
            
            # Suggest based on recommendations
            if 'recommendations' in strategy_evaluation:
                for rec in strategy_evaluation['recommendations']:
                    if 'optimize' in rec.lower():
                        suggestions.append(f"Implement optimization: {rec}")
            
        except Exception as e:
            logging.error(f"[{self.name}] Error generating planning optimizations: {e}")
            suggestions.append("Review planning optimization generation for errors")
        
        return suggestions
    
    def analyze_planning_patterns_for_goals(self) -> Dict[str, Any]:
        """
        Analyze planning patterns to identify opportunities for self-improvement goals.
        
        Returns:
            Analysis results with goal suggestions
        """
        analysis = {
            'pattern_insights': [],
            'improvement_opportunities': [],
            'suggested_goals': [],
            'confidence': 0.0
        }
        
        try:
            # Get planning-related memories
            planning_memories = self._get_planning_memories()
            
            if not planning_memories:
                analysis['suggested_goals'].append({
                    'goal_text': 'Increase planning activity to gather performance data',
                    'category': 'data_collection',
                    'priority': 'medium',
                    'rationale': 'Insufficient planning data for pattern analysis'
                })
                return analysis
            
            # Analyze success patterns
            success_patterns = self._analyze_planning_success_patterns(planning_memories)
            analysis['pattern_insights'].extend(success_patterns)
            
            # Analyze failure patterns
            failure_patterns = self._analyze_planning_failure_patterns(planning_memories)
            analysis['pattern_insights'].extend(failure_patterns)
            
            # Generate improvement opportunities
            opportunities = self._identify_planning_improvement_opportunities(success_patterns, failure_patterns)
            analysis['improvement_opportunities'] = opportunities
            
            # Convert opportunities to goals
            for opportunity in opportunities:
                goal = self._convert_opportunity_to_goal(opportunity)
                if goal:
                    analysis['suggested_goals'].append(goal)
            
            # Calculate confidence based on data quality
            analysis['confidence'] = min(1.0, len(planning_memories) / 20)
            
        except Exception as e:
            logging.error(f"[{self.name}] Error analyzing planning patterns: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _analyze_planning_performance(self) -> Dict[str, Any]:
        """Analyze current planning performance metrics."""
        performance = {
            'success_rate': 0.0,
            'average_confidence': 0.0,
            'common_failure_types': [],
            'performance_trends': {},
            'capability_gaps': []
        }
        
        try:
            # Get planning memories
            planning_memories = self._get_planning_memories()
            
            if not planning_memories:
                performance['capability_gaps'].append('No planning execution history')
                return performance
            
            # Calculate success rate
            successful = sum(1 for m in planning_memories if 'success' in m.get('content', '').lower())
            total = len(planning_memories)
            performance['success_rate'] = successful / total if total > 0 else 0.0
            
            # Calculate average confidence
            confidences = [m.get('confidence', 0.5) for m in planning_memories]
            performance['average_confidence'] = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Identify common failure types
            failures = [m for m in planning_memories if 'fail' in m.get('content', '').lower()]
            failure_types = {}
            for failure in failures:
                content = failure.get('content', '').lower()
                if 'timeout' in content:
                    failure_types['timeout'] = failure_types.get('timeout', 0) + 1
                elif 'dependency' in content:
                    failure_types['dependency'] = failure_types.get('dependency', 0) + 1
                elif 'resource' in content:
                    failure_types['resource'] = failure_types.get('resource', 0) + 1
                else:
                    failure_types['other'] = failure_types.get('other', 0) + 1
            
            performance['common_failure_types'] = sorted(failure_types.items(), key=lambda x: x[1], reverse=True)
            
            # Identify capability gaps
            if performance['success_rate'] < 0.7:
                performance['capability_gaps'].append('Low overall success rate')
            if performance['average_confidence'] < 0.6:
                performance['capability_gaps'].append('Low confidence in planning decisions')
            if failure_types.get('timeout', 0) > 2:
                performance['capability_gaps'].append('Time estimation accuracy issues')
            if failure_types.get('dependency', 0) > 2:
                performance['capability_gaps'].append('Dependency management problems')
        
        except Exception as e:
            logging.error(f"[{self.name}] Error analyzing planning performance: {e}")
            performance['error'] = str(e)
        
        return performance
    
    def _generate_goals_from_performance_gaps(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate goals to address performance gaps."""
        goals = []
        
        for gap in analysis.get('capability_gaps', []):
            if 'success rate' in gap.lower():
                goals.append({
                    'goal_text': 'Improve planning strategy success rate through better validation',
                    'category': 'performance_improvement',
                    'priority': 'high',
                    'rationale': 'Address low planning success rate identified in performance analysis',
                    'target_metric': 'success_rate',
                    'target_value': 0.8
                })
            
            elif 'confidence' in gap.lower():
                goals.append({
                    'goal_text': 'Enhance planning confidence through better data analysis',
                    'category': 'capability_enhancement',
                    'priority': 'medium',
                    'rationale': 'Low confidence indicates need for better planning foundations',
                    'target_metric': 'average_confidence',
                    'target_value': 0.75
                })
            
            elif 'time estimation' in gap.lower():
                goals.append({
                    'goal_text': 'Develop historical data-based time estimation model',
                    'category': 'algorithm_improvement',
                    'priority': 'high',
                    'rationale': 'Time estimation errors causing planning failures',
                    'target_metric': 'timeout_failure_rate',
                    'target_value': 0.1
                })
            
            elif 'dependency' in gap.lower():
                goals.append({
                    'goal_text': 'Implement comprehensive dependency validation system',
                    'category': 'system_enhancement',
                    'priority': 'high',
                    'rationale': 'Dependency failures are a significant cause of planning issues',
                    'target_metric': 'dependency_failure_rate',
                    'target_value': 0.05
                })
        
        return goals
    
    def _generate_optimization_goals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate goals for performance optimization."""
        goals = []
        
        # Always suggest some optimization goals
        optimization_opportunities = [
            {
                'goal_text': 'Optimize strategy scoring algorithm for better selection accuracy',
                'category': 'algorithm_optimization',
                'priority': 'medium',
                'rationale': 'Improved scoring leads to better strategy selection'
            },
            {
                'goal_text': 'Implement parallel execution optimization in suitable plans',
                'category': 'performance_optimization',
                'priority': 'medium',
                'rationale': 'Parallel execution can significantly reduce completion times'
            },
            {
                'goal_text': 'Develop adaptive re-planning triggers based on context changes',
                'category': 'adaptive_improvement',
                'priority': 'medium',
                'rationale': 'Better adaptation to changing conditions improves success rates'
            }
        ]
        
        # Filter based on current performance
        success_rate = analysis.get('success_rate', 0.5)
        if success_rate > 0.7:
            # High success rate - focus on efficiency optimizations
            goals = [g for g in optimization_opportunities if 'optimization' in g['category']]
        else:
            # Low success rate - focus on fundamental improvements
            goals = [g for g in optimization_opportunities if 'improvement' in g['category']]
        
        return goals
    
    def _generate_capability_expansion_goals(self) -> List[Dict[str, Any]]:
        """Generate goals for expanding planning capabilities."""
        expansion_goals = [
            {
                'goal_text': 'Implement multi-objective optimization in strategy selection',
                'category': 'capability_expansion',
                'priority': 'low',
                'rationale': 'Multi-objective optimization enables more sophisticated planning'
            },
            {
                'goal_text': 'Develop planning templates for common goal patterns',
                'category': 'efficiency_improvement',
                'priority': 'medium',
                'rationale': 'Templates can speed up planning for recurring goal types'
            },
            {
                'goal_text': 'Integrate external resource availability APIs for better planning',
                'category': 'integration_improvement',
                'priority': 'low',
                'rationale': 'Real-time resource data improves planning accuracy'
            }
        ]
        
        return expansion_goals
    
    def _score_planning_goal(self, goal: Dict[str, Any]) -> float:
        """Score a planning improvement goal."""
        score = 0.5  # Base score
        
        # Priority impact
        priority = goal.get('priority', 'medium')
        if priority == 'high':
            score += 0.3
        elif priority == 'medium':
            score += 0.1
        
        # Category impact
        category = goal.get('category', '')
        if 'improvement' in category or 'enhancement' in category:
            score += 0.2
        elif 'optimization' in category:
            score += 0.1
        
        # Target metric impact
        if 'target_metric' in goal:
            score += 0.1
        
        return min(1.0, score)
    
    def _get_planning_memories(self) -> List[Dict[str, Any]]:
        """Get planning-related memories for analysis."""
        memories = []
        
        try:
            if hasattr(self, 'memory_log') and self.memory_log:
                # Search for planning-related memories
                planning_facts = self.memory_log.search_facts(subject="planning")
                strategy_facts = self.memory_log.search_facts(subject="strategy")
                goal_facts = self.memory_log.search_facts(subject="goal")
                
                for fact in (planning_facts + strategy_facts + goal_facts)[:50]:  # Limit to recent
                    memories.append({
                        'subject': fact.subject,
                        'predicate': fact.predicate,
                        'object': fact.object,
                        'content': fact.object,
                        'confidence': getattr(fact, 'confidence', 0.5),
                        'timestamp': getattr(fact, 'timestamp', None)
                    })
        
        except Exception as e:
            logging.error(f"[{self.name}] Error getting planning memories: {e}")
        
        return memories
    
    def _analyze_planning_success_patterns(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Analyze patterns in successful planning."""
        patterns = []
        
        success_memories = [m for m in memories if any(
            keyword in m.get('content', '').lower() 
            for keyword in ['success', 'completed', 'achieved', 'effective']
        )]
        
        # Look for common success factors
        if len(success_memories) > 3:
            patterns.append("Multiple successful planning instances recorded")
            
            # Analyze content for patterns
            content_words = []
            for memory in success_memories:
                content_words.extend(memory.get('content', '').lower().split())
            
            word_counts = Counter(content_words)
            common_words = [word for word, count in word_counts.most_common(10) 
                          if count > 1 and len(word) > 3]
            
            if 'parallel' in common_words:
                patterns.append("Parallel execution appears in successful plans")
            if 'checkpoint' in common_words:
                patterns.append("Checkpoints are associated with success")
            if 'fallback' in common_words:
                patterns.append("Fallback options improve success rates")
        
        return patterns
    
    def _analyze_planning_failure_patterns(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Analyze patterns in planning failures."""
        patterns = []
        
        failure_memories = [m for m in memories if any(
            keyword in m.get('content', '').lower() 
            for keyword in ['fail', 'error', 'timeout', 'blocked']
        )]
        
        if len(failure_memories) > 2:
            patterns.append("Multiple planning failures need attention")
            
            # Analyze failure content
            content_words = []
            for memory in failure_memories:
                content_words.extend(memory.get('content', '').lower().split())
            
            word_counts = Counter(content_words)
            common_words = [word for word, count in word_counts.most_common(10) 
                          if count > 1 and len(word) > 3]
            
            if 'timeout' in common_words:
                patterns.append("Timeout is a recurring failure mode")
            if 'dependency' in common_words:
                patterns.append("Dependency issues cause failures")
            if 'resource' in common_words:
                patterns.append("Resource availability affects success")
        
        return patterns
    
    def _identify_planning_improvement_opportunities(self, success_patterns: List[str], failure_patterns: List[str]) -> List[str]:
        """Identify specific improvement opportunities."""
        opportunities = []
        
        # Opportunities from failure patterns
        for pattern in failure_patterns:
            if 'timeout' in pattern.lower():
                opportunities.append('Improve time estimation accuracy')
            elif 'dependency' in pattern.lower():
                opportunities.append('Enhance dependency management')
            elif 'resource' in pattern.lower():
                opportunities.append('Better resource planning and validation')
        
        # Opportunities to amplify success patterns
        for pattern in success_patterns:
            if 'parallel' in pattern.lower():
                opportunities.append('Increase use of parallel execution')
            elif 'checkpoint' in pattern.lower():
                opportunities.append('Standardize checkpoint implementation')
            elif 'fallback' in pattern.lower():
                opportunities.append('Mandate fallback options for critical steps')
        
        # General improvement opportunities
        if not success_patterns:
            opportunities.append('Establish success pattern tracking')
        if not failure_patterns:
            opportunities.append('Implement failure mode analysis')
        
        return opportunities
    
    def _convert_opportunity_to_goal(self, opportunity: str) -> Optional[Dict[str, Any]]:
        """Convert an improvement opportunity into a specific goal."""
        if 'time estimation' in opportunity.lower():
            return {
                'goal_text': 'Implement machine learning-based time estimation',
                'category': 'algorithm_improvement',
                'priority': 'high',
                'rationale': f'Address opportunity: {opportunity}'
            }
        
        elif 'dependency management' in opportunity.lower():
            return {
                'goal_text': 'Create comprehensive dependency validation framework',
                'category': 'system_enhancement',
                'priority': 'high',
                'rationale': f'Address opportunity: {opportunity}'
            }
        
        elif 'resource planning' in opportunity.lower():
            return {
                'goal_text': 'Develop predictive resource availability system',
                'category': 'capability_enhancement',
                'priority': 'medium',
                'rationale': f'Address opportunity: {opportunity}'
            }
        
        elif 'parallel execution' in opportunity.lower():
            return {
                'goal_text': 'Optimize automatic parallelization detection',
                'category': 'performance_optimization',
                'priority': 'medium',
                'rationale': f'Amplify success pattern: {opportunity}'
            }
        
        elif 'checkpoint' in opportunity.lower():
            return {
                'goal_text': 'Standardize checkpoint implementation across all plans',
                'category': 'consistency_improvement',
                'priority': 'medium',
                'rationale': f'Amplify success pattern: {opportunity}'
            }
        
        else:
            # Generic goal for unmatched opportunities
            return {
                'goal_text': f'Address planning improvement: {opportunity}',
                'category': 'general_improvement',
                'priority': 'medium',
                'rationale': f'Systematic improvement opportunity: {opportunity}'
            }
    
    def _generate_criterion_improvement_suggestion(self, criterion: str, score: float) -> Optional[str]:
        """Generate improvement suggestion for a low-scoring criterion."""
        if criterion == 'feasibility':
            return "Improve feasibility assessment by adding historical success rate analysis"
        elif criterion == 'complexity':
            return "Reduce plan complexity through better step decomposition"
        elif criterion == 'resource_efficiency':
            return "Optimize resource usage through better allocation algorithms"
        elif criterion == 'time_to_completion':
            return "Improve time estimates using historical execution data"
        elif criterion == 'success_probability':
            return "Enhance success probability calculation with more factors"
        elif criterion == 'risk_mitigation':
            return "Add more comprehensive risk assessment and mitigation strategies"
        else:
            return f"Improve {criterion} through systematic analysis and optimization"
    
    def _generate_weakness_improvement_suggestion(self, weakness: str) -> Optional[str]:
        """Generate improvement suggestion for a strategy weakness."""
        weakness_lower = weakness.lower()
        
        if 'complexity' in weakness_lower:
            return "Simplify strategy through step consolidation and dependency reduction"
        elif 'risk' in weakness_lower:
            return "Add additional risk mitigation measures and fallback options"
        elif 'time' in weakness_lower:
            return "Refine time estimates based on similar past executions"
        elif 'resource' in weakness_lower:
            return "Optimize resource requirements through efficiency analysis"
        else:
            return f"Address strategy weakness: {weakness}"