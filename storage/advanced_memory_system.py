#!/usr/bin/env python3
"""
🧠 Advanced Memory System for MeRNSTA

Integrates all advanced cognitive features:
- 🪢 Contradiction Clustering
- 🧬 Belief Consolidation  
- 🧠 Memory Graphs
- 🛡️ Confabulation Filtering
- 🤖 Meta-Cognition Agent

This creates a truly self-aware, adaptive memory system that not only stores facts
but actively monitors its own cognitive health and suggests improvements.
"""

import logging
import time
import json
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass

from .enhanced_memory_system import EnhancedMemorySystem
from .enhanced_triplet_extractor import EnhancedTripletFact

# Import all advanced cognitive systems
try:
    from .contradiction_clustering import ContradictionClusteringSystem
    from .belief_consolidation import BeliefConsolidationSystem
    from .memory_graphs import MemoryGraphSystem
    from .confabulation_filtering import ConfabulationFilteringSystem
    from .meta_cognition_agent import MetaCognitionAgent
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    ADVANCED_FEATURES_AVAILABLE = False
    logging.warning(f"Advanced cognitive features not available: {e}")


@dataclass
class CognitiveState:
    """Represents the current cognitive state of the system."""
    contradiction_clusters: Dict
    belief_patterns: Dict
    memory_graph: Optional[Any]
    cognitive_health: Optional[Any]
    priority_meta_goals: List
    confabulation_history: List
    last_cognitive_scan: float
    system_insights: List[str]


class AdvancedMemorySystem(EnhancedMemorySystem):
    """
    Enhanced memory system with advanced cognitive capabilities.
    Provides self-awareness, contradiction resolution, and meta-cognitive insights.
    """
    
    def __init__(self):
        super().__init__()
        
        if ADVANCED_FEATURES_AVAILABLE:
            self.contradiction_clustering = ContradictionClusteringSystem()
            self.belief_consolidation = BeliefConsolidationSystem()
            self.memory_graphs = MemoryGraphSystem()
            self.confabulation_filter = ConfabulationFilteringSystem()
            self.meta_cognition = MetaCognitionAgent()
            
            self.cognitive_scan_interval = 3600  # 1 hour
            self.last_full_scan = 0.0
            self.confabulation_assessments = []
            
            print("[AdvancedMemory] All cognitive subsystems initialized")
        else:
            print("[AdvancedMemory] Running in basic mode - advanced features unavailable")
    
    def process_input_with_cognition(self, text: str, user_profile_id: str = None, 
                                   session_id: str = None) -> Dict:
        """
        Process input with full cognitive awareness and filtering.
        """
        # First process normally
        base_result = super().process_input(text, user_profile_id, session_id)
        
        if not ADVANCED_FEATURES_AVAILABLE:
            return base_result
        
        # Check if it's time for a cognitive scan
        if time.time() - self.last_full_scan > self.cognitive_scan_interval:
            self._perform_cognitive_scan(user_profile_id, session_id)
        
        # If this was a query, apply confabulation filtering
        if 'response' in base_result and base_result.get('response'):
            filtered_result = self._apply_confabulation_filtering(
                base_result['response'], text, user_profile_id, session_id
            )
            base_result.update(filtered_result)
        
        # Add cognitive insights
        base_result['cognitive_insights'] = self._get_current_insights()
        
        return base_result
    
    def _perform_cognitive_scan(self, user_profile_id: str = None, session_id: str = None):
        """Perform a comprehensive cognitive scan and update all subsystems."""
        print("[AdvancedMemory] Performing comprehensive cognitive scan...")
        
        # Get all relevant facts
        facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
        
        # 1. Analyze contradiction clusters
        contradiction_clusters = self.contradiction_clustering.analyze_contradictions(facts)
        
        # 2. Analyze belief consolidation
        belief_patterns = self.belief_consolidation.analyze_belief_stability(facts)
        
        # 3. Build memory graph
        memory_graph = self.memory_graphs.build_memory_graph(
            facts, contradiction_clusters, belief_patterns
        )
        
        # 4. Perform meta-cognitive analysis
        cognitive_health = self.meta_cognition.perform_cognitive_scan(
            facts, contradiction_clusters, belief_patterns
        )
        
        self.last_full_scan = time.time()
        
        print("[AdvancedMemory] Cognitive scan completed")
        print(f"  → Found {len(contradiction_clusters)} contradiction clusters")
        print(f"  → Identified {len(belief_patterns)} belief patterns") 
        print(f"  → Cognitive health score: {cognitive_health.overall_score:.2f}")
        print(f"  → Generated {len(self.meta_cognition.get_priority_goals())} priority meta-goals")
    
    def _apply_confabulation_filtering(self, response: str, query: str,
                                     user_profile_id: str = None, 
                                     session_id: str = None) -> Dict:
        """Apply confabulation filtering to response."""
        facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
        
        # Get contradiction history
        contradiction_history = {
            'reliability_score': 0.7  # Default - could be calculated from history
        }
        
        # Assess response reliability
        assessment = self.confabulation_filter.assess_response_reliability(
            response, query, facts, contradiction_history
        )
        
        # Store assessment
        self.confabulation_assessments.append(assessment)
        
        # Keep only recent assessments
        cutoff_time = time.time() - (24 * 3600)  # 24 hours
        self.confabulation_assessments = [
            a for a in self.confabulation_assessments 
            if a.reliability_factors.get('timestamp', time.time()) > cutoff_time
        ]
        
        return {
            'original_response': response,
            'filtered_response': assessment.filtered_response,
            'confabulation_assessment': {
                'confidence_score': assessment.confidence_score,
                'confabulation_risk': assessment.confabulation_risk,
                'action_taken': assessment.action_taken,
                'supporting_facts_count': len(assessment.supporting_facts),
                'contradicting_facts_count': len(assessment.contradicting_facts)
            }
        }
    
    def _get_current_insights(self) -> List[str]:
        """Get current cognitive insights and recommendations."""
        insights = []
        
        # Get meta-cognitive suggestions
        meta_suggestions = self.meta_cognition.generate_self_improvement_suggestions()
        insights.extend(meta_suggestions[:3])  # Top 3 suggestions
        
        # Get contradiction clustering insights
        cluster_insights = self.contradiction_clustering.suggest_cluster_insights()
        insights.extend(cluster_insights[:2])  # Top 2 insights
        
        # Get belief consolidation suggestions
        consolidation_suggestions = self.belief_consolidation.suggest_consolidation_actions()
        insights.extend(consolidation_suggestions[:2])  # Top 2 suggestions
        
        return insights
    
    def get_cognitive_dashboard(self, user_profile_id: str = None, 
                              session_id: str = None) -> Dict:
        """Get comprehensive cognitive state dashboard."""
        if not ADVANCED_FEATURES_AVAILABLE:
            return {'error': 'Advanced cognitive features not available'}
        
        facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
        
        # Force a scan if we haven't done one recently
        if time.time() - self.last_full_scan > 300:  # 5 minutes
            self._perform_cognitive_scan(user_profile_id, session_id)
        
        dashboard = {
            'system_overview': {
                'total_facts': len(facts),
                'last_cognitive_scan': self.last_full_scan,
                'advanced_features_active': True,
                'scan_interval_hours': self.cognitive_scan_interval / 3600
            },
            
            'contradiction_analysis': self.contradiction_clustering.get_cluster_summary(),
            
            'belief_consolidation': self.belief_consolidation.get_consolidation_summary(),
            
            'memory_graph': self.memory_graphs.get_graph_summary(),
            
            'meta_cognition': self.meta_cognition.get_cognitive_summary(),
            
            'confabulation_filtering': self.confabulation_filter.get_filtering_summary(
                self.confabulation_assessments
            ),
            
            'priority_insights': self._get_current_insights(),
            
            'system_health': {
                'overall_status': self._calculate_overall_health(),
                'critical_issues': self._identify_critical_issues(),
                'recommendations': self._get_priority_recommendations()
            }
        }
        
        return dashboard
    
    def _calculate_overall_health(self) -> str:
        """Calculate overall system health status."""
        if not hasattr(self.meta_cognition, 'assessment_history') or not self.meta_cognition.assessment_history:
            return "unknown"
        
        latest_health = self.meta_cognition.assessment_history[-1]
        health_score = latest_health.overall_score
        
        if health_score >= 0.8:
            return "excellent"
        elif health_score >= 0.6:
            return "good"
        elif health_score >= 0.4:
            return "fair"
        else:
            return "needs_attention"
    
    def _identify_critical_issues(self) -> List[str]:
        """Identify critical issues that need immediate attention."""
        critical_issues = []
        
        # High priority meta-goals
        high_priority_goals = [g for g in self.meta_cognition.get_priority_goals() if g.priority >= 8]
        if len(high_priority_goals) > 3:
            critical_issues.append(f"Multiple critical meta-goals pending ({len(high_priority_goals)})")
        
        # High confabulation risk responses
        recent_high_risk = [a for a in self.confabulation_assessments[-10:] 
                           if a.confabulation_risk > 0.8]
        if len(recent_high_risk) > 2:
            critical_issues.append("High rate of confabulation risk in recent responses")
        
        # Cognitive health issues
        if hasattr(self.meta_cognition, 'assessment_history') and self.meta_cognition.assessment_history:
            latest_health = self.meta_cognition.assessment_history[-1]
            if len(latest_health.issues_detected) > 5:
                critical_issues.append("Multiple cognitive health issues detected")
        
        return critical_issues
    
    def _get_priority_recommendations(self) -> List[str]:
        """Get priority recommendations for system improvement."""
        recommendations = []
        
        # From meta-cognition
        if hasattr(self.meta_cognition, 'assessment_history') and self.meta_cognition.assessment_history:
            latest_health = self.meta_cognition.assessment_history[-1]
            recommendations.extend(latest_health.recommendations[:2])
        
        # Critical meta-goals
        critical_goals = [g for g in self.meta_cognition.get_priority_goals() if g.priority >= 8]
        for goal in critical_goals[:2]:
            recommendations.append(f"URGENT: {goal.description}")
        
        # Confabulation improvements
        if len(self.confabulation_assessments) > 5:
            recent_assessments = self.confabulation_assessments[-10:]
            avg_confidence = sum(a.confidence_score for a in recent_assessments) / len(recent_assessments)
            if avg_confidence < 0.6:
                recommendations.append("Consider improving response confidence through fact verification")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def export_cognitive_state(self, output_file: str, user_profile_id: str = None,
                             session_id: str = None) -> bool:
        """Export complete cognitive state for analysis or backup."""
        if not ADVANCED_FEATURES_AVAILABLE:
            return False
        
        try:
            dashboard = self.get_cognitive_dashboard(user_profile_id, session_id)
            
            # Add detailed data
            facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
            
            export_data = {
                'export_metadata': {
                    'timestamp': time.time(),
                    'user_profile_id': user_profile_id,
                    'session_id': session_id,
                    'system_version': 'MeRNSTA Advanced v1.0'
                },
                'cognitive_dashboard': dashboard,
                'raw_facts': [
                    {
                        'id': f.id,
                        'subject': f.subject,
                        'predicate': f.predicate,
                        'object': f.object,
                        'confidence': f.confidence,
                        'timestamp': getattr(f, 'timestamp', None),
                        'contradiction': getattr(f, 'contradiction', False),
                        'volatility_score': getattr(f, 'volatility_score', 0.0)
                    }
                    for f in facts
                ],
                'meta_goals': [
                    {
                        'goal_id': g.goal_id,
                        'goal_type': g.goal_type,
                        'priority': g.priority,
                        'description': g.description,
                        'target_concept': g.target_concept,
                        'suggested_actions': g.suggested_actions,
                        'status': g.status,
                        'created_time': g.created_time
                    }
                    for g in self.meta_cognition.meta_goals.values()
                ],
                'recent_confabulation_assessments': [
                    {
                        'confidence_score': a.confidence_score,
                        'confabulation_risk': a.confabulation_risk,
                        'action_taken': a.action_taken,
                        'supporting_facts_count': len(a.supporting_facts),
                        'contradicting_facts_count': len(a.contradicting_facts)
                    }
                    for a in self.confabulation_assessments[-20:]  # Last 20
                ]
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"[AdvancedMemory] Cognitive state exported to {output_file}")
            return True
            
        except Exception as e:
            print(f"[AdvancedMemory] Failed to export cognitive state: {e}")
            return False
    
    def generate_memory_visualization(self, output_file: str = None, 
                                    user_profile_id: str = None,
                                    session_id: str = None) -> bool:
        """Generate a visual representation of the memory graph."""
        if not ADVANCED_FEATURES_AVAILABLE:
            return False
        
        facts = self.get_facts(user_profile_id=user_profile_id, session_id=session_id)
        
        # Get current analysis
        contradiction_clusters = self.contradiction_clustering.analyze_contradictions(facts)
        belief_patterns = self.belief_consolidation.analyze_belief_stability(facts)
        
        # Build and visualize graph
        memory_graph = self.memory_graphs.build_memory_graph(
            facts, contradiction_clusters, belief_patterns
        )
        
        if memory_graph:
            return self.memory_graphs.visualize_graph(output_file, 
                                                    show_labels=True,
                                                    highlight_contradictions=True)
        
        return False 