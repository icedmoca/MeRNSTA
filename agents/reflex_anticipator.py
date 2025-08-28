#!/usr/bin/env python3
"""
Reflex Anticipator for MeRNSTA

This module implements anticipatory reflex logic that checks for predictive goals
before triggering drift cycles, enabling proactive repair before symptoms appear.
"""

import sqlite3
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import defaultdict

from storage.causal_drift_predictor import CausalDriftPredictor, PredictiveGoal
from storage.reflex_compression import ReflexCompressor, ReflexTemplate
from storage.reflex_log import ReflexCycle, ReflexLogger


@dataclass
class AnticipatoryReflex:
    """Represents an anticipatory reflex triggered by predictive goals."""
    reflex_id: str
    predictive_goal_id: str
    template_id: Optional[str]
    strategy: str
    execution_result: Dict[str, Any]
    created_at: float = 0.0
    executed_at: Optional[float] = None
    success: bool = False
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnticipatoryReflex':
        """Create from dictionary."""
        return cls(**data)


class ReflexAnticipator:
    """
    Reflex Anticipator that enables proactive repair before drift symptoms appear.
    
    Features:
    - Checks for predictive goals before triggering drift cycles
    - Uses reflex templates for immediate execution
    - Skips LLM when pattern matches are found
    - Tracks anticipatory reflex effectiveness
    """
    
    def __init__(self, db_path: str = "causal_predictions.db"):
        self.db_path = db_path
        self.predictor = CausalDriftPredictor(db_path)
        self.compressor = ReflexCompressor()
        self.reflex_logger = ReflexLogger()
        
        # Anticipatory parameters
        self.template_match_threshold = 0.8  # Minimum similarity for template match
        self.max_anticipatory_reflexes = 5  # Maximum reflexes per prediction window
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize anticipatory reflex database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anticipatory_reflexes (
                reflex_id TEXT PRIMARY KEY,
                predictive_goal_id TEXT NOT NULL,
                template_id TEXT,
                strategy TEXT NOT NULL,
                execution_result TEXT NOT NULL,
                created_at REAL NOT NULL,
                executed_at REAL,
                success BOOLEAN DEFAULT FALSE,
                INDEX(predictive_goal_id),
                INDEX(success),
                INDEX(created_at)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def check_for_anticipatory_reflex(self, token_id: Optional[int] = None,
                                     cluster_id: Optional[str] = None,
                                     drift_score: float = 0.0) -> Optional[AnticipatoryReflex]:
        """
        Check if an anticipatory reflex should be triggered.
        
        Args:
            token_id: Optional token ID for context
            cluster_id: Optional cluster ID for context
            drift_score: Current drift score
            
        Returns:
            AnticipatoryReflex if triggered, None otherwise
        """
        # Get pending predictive goals
        pending_goals = self.predictor.get_pending_predictive_goals(limit=10)
        
        # Filter goals by context
        relevant_goals = self._filter_relevant_goals(pending_goals, token_id, cluster_id, drift_score)
        
        if not relevant_goals:
            return None
        
        # Find the most urgent goal
        urgent_goal = max(relevant_goals, key=lambda g: g.urgency)
        
        # Check if we have a matching template
        matching_template = self._find_matching_template(urgent_goal)
        
        if matching_template:
            # Execute anticipatory reflex using template
            return self._execute_anticipatory_reflex(urgent_goal, matching_template)
        else:
            # Generate new anticipatory reflex
            return self._generate_anticipatory_reflex(urgent_goal)
    
    def _filter_relevant_goals(self, goals: List[PredictiveGoal], 
                              token_id: Optional[int], 
                              cluster_id: Optional[str], 
                              drift_score: float) -> List[PredictiveGoal]:
        """Filter goals relevant to current context."""
        relevant_goals = []
        
        for goal in goals:
            # Check token match
            if token_id and goal.token_id == token_id:
                relevant_goals.append(goal)
                continue
            
            # Check cluster match
            if cluster_id and goal.cluster_id == cluster_id:
                relevant_goals.append(goal)
                continue
            
            # Check urgency threshold
            if goal.urgency > 0.7:
                relevant_goals.append(goal)
                continue
            
            # Check if drift score matches prediction
            if self._drift_matches_prediction(goal, drift_score):
                relevant_goals.append(goal)
        
        return relevant_goals
    
    def _drift_matches_prediction(self, goal: PredictiveGoal, drift_score: float) -> bool:
        """Check if current drift score matches the prediction."""
        # Extract predicted drift from source metrics
        source_metrics = goal.source_metrics
        
        if 'predicted' in source_metrics:
            predicted_drift = source_metrics['predicted']
            # Check if current drift is approaching predicted value
            return abs(drift_score - predicted_drift) < 0.2
        
        return False
    
    def _find_matching_template(self, goal: PredictiveGoal) -> Optional[ReflexTemplate]:
        """Find a matching reflex template for the goal."""
        # Get template suggestions based on goal characteristics
        suggestions = self.compressor.suggest_templates_for_drift(
            drift_score=0.8,  # Assume high drift for predictive goals
            token_id=goal.token_id,
            cluster_id=goal.cluster_id
        )
        
        if not suggestions:
            return None
        
        # Find best matching template
        best_template = None
        best_score = 0.0
        
        for template in suggestions:
            match_score = self._calculate_template_match_score(template, goal)
            if match_score > self.template_match_threshold and match_score > best_score:
                best_score = match_score
                best_template = template
        
        return best_template
    
    def _calculate_template_match_score(self, template: ReflexTemplate, goal: PredictiveGoal) -> float:
        """Calculate how well a template matches a predictive goal."""
        scores = []
        
        # Strategy match
        if template.strategy in goal.prediction_type:
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        # Goal pattern match
        if any(word in template.goal_pattern.lower() for word in goal.prediction_type.lower().split('_')):
            scores.append(0.8)
        else:
            scores.append(0.2)
        
        # Success rate weight
        scores.append(template.success_rate)
        
        # Usage count weight (normalized)
        usage_score = min(template.usage_count / 10.0, 1.0)
        scores.append(usage_score)
        
        return np.mean(scores)
    
    def _execute_anticipatory_reflex(self, goal: PredictiveGoal, 
                                    template: ReflexTemplate) -> AnticipatoryReflex:
        """Execute an anticipatory reflex using a template."""
        reflex_id = f"anticipatory_{goal.goal_id}_{int(time.time())}"
        
        # Mark goal as executed
        self.predictor.mark_goal_executed(goal.goal_id, reflex_id)
        
        # Execute template actions
        execution_result = self._execute_template_actions(template, goal)
        
        # Create anticipatory reflex record
        anticipatory_reflex = AnticipatoryReflex(
            reflex_id=reflex_id,
            predictive_goal_id=goal.goal_id,
            template_id=template.template_id,
            strategy=template.strategy,
            execution_result=execution_result,
            executed_at=time.time(),
            success=execution_result.get('success', False)
        )
        
        # Store anticipatory reflex
        self._store_anticipatory_reflex(anticipatory_reflex)
        
        # Increment template usage
        self.compressor.increment_usage(template.template_id)
        
        logging.info(f"[ReflexAnticipator] Executed anticipatory reflex {reflex_id} using template {template.template_id}")
        
        return anticipatory_reflex
    
    def _generate_anticipatory_reflex(self, goal: PredictiveGoal) -> AnticipatoryReflex:
        """Generate a new anticipatory reflex without a template."""
        reflex_id = f"anticipatory_{goal.goal_id}_{int(time.time())}"
        
        # Mark goal as executed
        self.predictor.mark_goal_executed(goal.goal_id, reflex_id)
        
        # Generate strategy based on prediction type
        strategy = self._determine_strategy_from_prediction(goal)
        
        # Execute strategy
        execution_result = self._execute_strategy(strategy, goal)
        
        # Create anticipatory reflex record
        anticipatory_reflex = AnticipatoryReflex(
            reflex_id=reflex_id,
            predictive_goal_id=goal.goal_id,
            template_id=None,
            strategy=strategy,
            execution_result=execution_result,
            executed_at=time.time(),
            success=execution_result.get('success', False)
        )
        
        # Store anticipatory reflex
        self._store_anticipatory_reflex(anticipatory_reflex)
        
        logging.info(f"[ReflexAnticipator] Generated anticipatory reflex {reflex_id} with strategy {strategy}")
        
        return anticipatory_reflex
    
    def _determine_strategy_from_prediction(self, goal: PredictiveGoal) -> str:
        """Determine strategy based on prediction type."""
        prediction_type = goal.prediction_type
        
        if 'contradiction' in prediction_type:
            return 'belief_clarification'
        elif 'coherence' in prediction_type:
            return 'cluster_reassessment'
        elif 'volatility' in prediction_type:
            return 'fact_consolidation'
        else:
            return 'belief_clarification'  # Default
    
    def _execute_template_actions(self, template: ReflexTemplate, goal: PredictiveGoal) -> Dict[str, Any]:
        """Execute actions from a reflex template."""
        try:
            # Simulate template execution
            actions = template.execution_pattern
            
            # Execute each action
            results = []
            for action in actions:
                result = self._execute_action(action, goal)
                results.append(result)
            
            # Determine overall success
            success = all(r.get('success', False) for r in results)
            
            return {
                'success': success,
                'actions_executed': len(actions),
                'action_results': results,
                'template_used': template.template_id,
                'execution_time': 0.1  # Simulated
            }
            
        except Exception as e:
            logging.error(f"[ReflexAnticipator] Error executing template: {e}")
            return {
                'success': False,
                'error': str(e),
                'actions_executed': 0
            }
    
    def _execute_strategy(self, strategy: str, goal: PredictiveGoal) -> Dict[str, Any]:
        """Execute a strategy for anticipatory reflex."""
        try:
            # Simulate strategy execution
            if strategy == 'belief_clarification':
                result = self._execute_belief_clarification(goal)
            elif strategy == 'cluster_reassessment':
                result = self._execute_cluster_reassessment(goal)
            elif strategy == 'fact_consolidation':
                result = self._execute_fact_consolidation(goal)
            else:
                result = {'success': False, 'error': 'Unknown strategy'}
            
            return {
                'success': result.get('success', False),
                'strategy': strategy,
                'result': result,
                'execution_time': 0.1  # Simulated
            }
            
        except Exception as e:
            logging.error(f"[ReflexAnticipator] Error executing strategy: {e}")
            return {
                'success': False,
                'error': str(e),
                'strategy': strategy
            }
    
    def _execute_action(self, action: str, goal: PredictiveGoal) -> Dict[str, Any]:
        """Execute a single action."""
        # Simulate action execution
        return {
            'action': action,
            'success': True,
            'result': f"Executed {action} for goal {goal.goal_id}"
        }
    
    def _execute_belief_clarification(self, goal: PredictiveGoal) -> Dict[str, Any]:
        """Execute belief clarification strategy."""
        return {
            'success': True,
            'actions': ['analyzed_beliefs', 'identified_conflicts', 'clarified_ambiguities'],
            'notes': f"Clarified beliefs for {goal.prediction_type}"
        }
    
    def _execute_cluster_reassessment(self, goal: PredictiveGoal) -> Dict[str, Any]:
        """Execute cluster reassessment strategy."""
        return {
            'success': True,
            'actions': ['analyzed_cluster', 'identified_instability', 'reorganized_concepts'],
            'notes': f"Reassessed cluster for {goal.prediction_type}"
        }
    
    def _execute_fact_consolidation(self, goal: PredictiveGoal) -> Dict[str, Any]:
        """Execute fact consolidation strategy."""
        return {
            'success': True,
            'actions': ['identified_facts', 'merged_similar', 'removed_redundant'],
            'notes': f"Consolidated facts for {goal.prediction_type}"
        }
    
    def _store_anticipatory_reflex(self, reflex: AnticipatoryReflex):
        """Store anticipatory reflex in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO anticipatory_reflexes 
            (reflex_id, predictive_goal_id, template_id, strategy, execution_result,
             created_at, executed_at, success)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            reflex.reflex_id, reflex.predictive_goal_id, reflex.template_id,
            reflex.strategy, json.dumps(reflex.execution_result),
            reflex.created_at, reflex.executed_at, reflex.success
        ))
        
        conn.commit()
        conn.close()
    
    def get_anticipatory_reflexes(self, limit: Optional[int] = None) -> List[AnticipatoryReflex]:
        """Get anticipatory reflexes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT reflex_id, predictive_goal_id, template_id, strategy, execution_result,
                   created_at, executed_at, success
            FROM anticipatory_reflexes 
            ORDER BY created_at DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        
        reflexes = []
        for row in cursor.fetchall():
            data = {
                'reflex_id': row[0],
                'predictive_goal_id': row[1],
                'template_id': row[2],
                'strategy': row[3],
                'execution_result': json.loads(row[4]),
                'created_at': row[5],
                'executed_at': row[6],
                'success': bool(row[7])
            }
            reflexes.append(AnticipatoryReflex.from_dict(data))
        
        conn.close()
        return reflexes
    
    def get_anticipatory_statistics(self) -> Dict[str, Any]:
        """Get statistics about anticipatory reflexes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total anticipatory reflexes
        cursor.execute("SELECT COUNT(*) FROM anticipatory_reflexes")
        total_reflexes = cursor.fetchone()[0]
        
        # Successful reflexes
        cursor.execute("SELECT COUNT(*) FROM anticipatory_reflexes WHERE success = TRUE")
        successful_reflexes = cursor.fetchone()[0]
        
        # Reflexes by strategy
        cursor.execute("""
            SELECT strategy, COUNT(*) 
            FROM anticipatory_reflexes 
            GROUP BY strategy
        """)
        by_strategy = dict(cursor.fetchall())
        
        # Reflexes with templates
        cursor.execute("SELECT COUNT(*) FROM anticipatory_reflexes WHERE template_id IS NOT NULL")
        template_reflexes = cursor.fetchone()[0]
        
        # Recent reflexes (last 24 hours)
        cursor.execute("""
            SELECT COUNT(*) FROM anticipatory_reflexes 
            WHERE created_at > ?
        """, (time.time() - 24 * 3600,))
        recent_reflexes = cursor.fetchone()[0]
        
        conn.close()
        
        success_rate = successful_reflexes / total_reflexes if total_reflexes > 0 else 0.0
        
        return {
            'total_reflexes': total_reflexes,
            'successful_reflexes': successful_reflexes,
            'success_rate': success_rate,
            'by_strategy': by_strategy,
            'template_reflexes': template_reflexes,
            'recent_reflexes_24h': recent_reflexes
        }
    
    def record_prediction_outcome(self, goal_id: str, actual_outcome: Dict[str, Any]):
        """Record the outcome of a predictive goal."""
        # Find the anticipatory reflex for this goal
        reflexes = self.get_anticipatory_reflexes()
        matching_reflex = None
        
        for reflex in reflexes:
            if reflex.predictive_goal_id == goal_id:
                matching_reflex = reflex
                break
        
        if matching_reflex:
            # Calculate accuracy
            predicted_success = matching_reflex.success
            actual_success = actual_outcome.get('success', False)
            accuracy = 1.0 if predicted_success == actual_success else 0.0
            
            # Record outcome
            self.predictor.record_prediction_outcome(goal_id, accuracy, accuracy)
            
            # Mark goal as completed
            self.predictor.mark_goal_completed(goal_id, actual_success)
            
            logging.info(f"[ReflexAnticipator] Recorded outcome for goal {goal_id}: accuracy={accuracy}")
    
    def should_skip_llm(self, token_id: Optional[int] = None,
                       cluster_id: Optional[str] = None,
                       drift_score: float = 0.0) -> bool:
        """
        Check if LLM should be skipped due to anticipatory reflex.
        
        Returns:
            True if anticipatory reflex was triggered, False otherwise
        """
        anticipatory_reflex = self.check_for_anticipatory_reflex(token_id, cluster_id, drift_score)
        return anticipatory_reflex is not None 