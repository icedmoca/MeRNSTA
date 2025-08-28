#!/usr/bin/env python3
"""
Causal Audit Dashboard for MeRNSTA

This module provides a comprehensive dashboard for monitoring predictive causal modeling,
hypothesis generation, and anticipatory reflex systems.
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
from agents.hypothesis_generator import HypothesisGeneratorAgent, Hypothesis
from agents.reflex_anticipator import ReflexAnticipator, AnticipatoryReflex


@dataclass
class DashboardMetrics:
    """Represents dashboard metrics for causal modeling."""
    timestamp: float
    total_predictions: int
    pending_predictions: int
    total_hypotheses: int
    open_hypotheses: int
    total_anticipatory_reflexes: int
    successful_reflexes: int
    prediction_accuracy: float
    hypothesis_confirmation_rate: float
    reflex_success_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DashboardMetrics':
        """Create from dictionary."""
        return cls(**data)


class CausalAuditDashboard:
    """
    Causal Audit Dashboard for monitoring predictive causal modeling.
    
    Features:
    - Shows predicted upcoming drifts
    - Displays active hypotheses and their statuses
    - Tracks anticipatory reflex templates triggered
    - Provides timeline of predicted vs actual outcomes
    - Offers comprehensive system metrics
    """
    
    def __init__(self, db_path: str = "causal_predictions.db"):
        self.db_path = db_path
        self.predictor = CausalDriftPredictor(db_path)
        self.hypothesis_generator = HypothesisGeneratorAgent(db_path)
        self.reflex_anticipator = ReflexAnticipator(db_path)
        
        # Dashboard parameters
        self.metrics_history_size = 100  # Number of historical metrics to keep
        self.update_interval = 300  # Update metrics every 5 minutes
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize dashboard database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Dashboard metrics history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dashboard_metrics (
                timestamp REAL PRIMARY KEY,
                total_predictions INTEGER NOT NULL,
                pending_predictions INTEGER NOT NULL,
                total_hypotheses INTEGER NOT NULL,
                open_hypotheses INTEGER NOT NULL,
                total_anticipatory_reflexes INTEGER NOT NULL,
                successful_reflexes INTEGER NOT NULL,
                prediction_accuracy REAL NOT NULL,
                hypothesis_confirmation_rate REAL NOT NULL,
                reflex_success_rate REAL NOT NULL,
                INDEX(timestamp)
            )
        """)
        
        # Predicted vs actual outcomes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS outcome_timeline (
                timeline_id TEXT PRIMARY KEY,
                prediction_id TEXT NOT NULL,
                predicted_outcome TEXT NOT NULL,
                actual_outcome TEXT,
                accuracy REAL,
                timestamp REAL NOT NULL,
                status TEXT DEFAULT 'pending',  # pending, confirmed, failed
                INDEX(prediction_id),
                INDEX(timestamp),
                INDEX(status)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """
        Generate comprehensive dashboard data.
        
        Returns:
            Dictionary containing all dashboard information
        """
        # Get current metrics
        current_metrics = self._get_current_metrics()
        
        # Get predicted upcoming drifts
        upcoming_drifts = self._get_upcoming_drifts()
        
        # Get active hypotheses
        active_hypotheses = self._get_active_hypotheses()
        
        # Get recent anticipatory reflexes
        recent_reflexes = self._get_recent_anticipatory_reflexes()
        
        # Get outcome timeline
        outcome_timeline = self._get_outcome_timeline()
        
        # Get system statistics
        system_stats = self._get_system_statistics()
        
        return {
            'timestamp': time.time(),
            'current_metrics': current_metrics,
            'upcoming_drifts': upcoming_drifts,
            'active_hypotheses': active_hypotheses,
            'recent_anticipatory_reflexes': recent_reflexes,
            'outcome_timeline': outcome_timeline,
            'system_statistics': system_stats
        }
    
    def _get_current_metrics(self) -> DashboardMetrics:
        """Get current dashboard metrics."""
        # Get prediction statistics
        pred_stats = self.predictor.get_prediction_statistics()
        
        # Get hypothesis statistics
        hyp_stats = self.hypothesis_generator.get_hypothesis_statistics()
        
        # Get anticipatory reflex statistics
        reflex_stats = self.reflex_anticipator.get_anticipatory_statistics()
        
        metrics = DashboardMetrics(
            timestamp=time.time(),
            total_predictions=pred_stats['total_predictions'],
            pending_predictions=pred_stats['pending_predictions'],
            total_hypotheses=hyp_stats['total_hypotheses'],
            open_hypotheses=hyp_stats['by_status'].get('open', 0),
            total_anticipatory_reflexes=reflex_stats['total_reflexes'],
            successful_reflexes=reflex_stats['successful_reflexes'],
            prediction_accuracy=pred_stats['average_accuracy'],
            hypothesis_confirmation_rate=hyp_stats['confirmation_rate'],
            reflex_success_rate=reflex_stats['success_rate']
        )
        
        # Store metrics for history
        self._store_dashboard_metrics(metrics)
        
        return metrics
    
    def _get_upcoming_drifts(self) -> List[Dict[str, Any]]:
        """Get predicted upcoming drifts."""
        pending_goals = self.predictor.get_pending_predictive_goals(limit=10)
        
        upcoming_drifts = []
        for goal in pending_goals:
            drift_info = {
                'goal_id': goal.goal_id,
                'token_id': goal.token_id,
                'cluster_id': goal.cluster_id,
                'prediction_type': goal.prediction_type,
                'predicted_outcome': goal.predicted_outcome,
                'probability': goal.probability,
                'urgency': goal.urgency,
                'created_at': goal.created_at,
                'time_until_predicted': self._calculate_time_until_predicted(goal)
            }
            upcoming_drifts.append(drift_info)
        
        # Sort by urgency and time
        upcoming_drifts.sort(key=lambda x: (x['urgency'], -x['time_until_predicted']), reverse=True)
        
        return upcoming_drifts
    
    def _calculate_time_until_predicted(self, goal: PredictiveGoal) -> float:
        """Calculate time until predicted outcome (in hours)."""
        # This is a simplified calculation
        # In practice, you'd extract the predicted time from the goal
        time_since_creation = time.time() - goal.created_at
        return max(0.0, 24.0 - (time_since_creation / 3600))  # Assume 24-hour window
    
    def _get_active_hypotheses(self) -> List[Dict[str, Any]]:
        """Get active hypotheses with their status."""
        open_hypotheses = self.hypothesis_generator.get_open_hypotheses(limit=15)
        
        active_hypotheses = []
        for hypothesis in open_hypotheses:
            hyp_info = {
                'hypothesis_id': hypothesis.hypothesis_id,
                'cause_token': hypothesis.cause_token,
                'predicted_outcome': hypothesis.predicted_outcome,
                'probability': hypothesis.probability,
                'confidence_score': hypothesis.confidence_score,
                'hypothesis_type': hypothesis.hypothesis_type,
                'supporting_evidence': hypothesis.supporting_evidence,
                'created_at': hypothesis.created_at,
                'reflex_cycle_id': hypothesis.reflex_cycle_id,
                'drift_goal_id': hypothesis.drift_goal_id
            }
            active_hypotheses.append(hyp_info)
        
        # Sort by confidence score
        active_hypotheses.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return active_hypotheses
    
    def _get_recent_anticipatory_reflexes(self) -> List[Dict[str, Any]]:
        """Get recent anticipatory reflexes."""
        recent_reflexes = self.reflex_anticipator.get_anticipatory_reflexes(limit=10)
        
        reflex_data = []
        for reflex in recent_reflexes:
            reflex_info = {
                'reflex_id': reflex.reflex_id,
                'predictive_goal_id': reflex.predictive_goal_id,
                'template_id': reflex.template_id,
                'strategy': reflex.strategy,
                'success': reflex.success,
                'created_at': reflex.created_at,
                'executed_at': reflex.executed_at,
                'execution_time': reflex.executed_at - reflex.created_at if reflex.executed_at else None
            }
            reflex_data.append(reflex_info)
        
        # Sort by creation time
        reflex_data.sort(key=lambda x: x['created_at'], reverse=True)
        
        return reflex_data
    
    def _get_outcome_timeline(self) -> List[Dict[str, Any]]:
        """Get timeline of predicted vs actual outcomes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timeline_id, prediction_id, predicted_outcome, actual_outcome,
                   accuracy, timestamp, status
            FROM outcome_timeline 
            ORDER BY timestamp DESC
            LIMIT 20
        """)
        
        timeline = []
        for row in cursor.fetchall():
            timeline.append({
                'timeline_id': row[0],
                'prediction_id': row[1],
                'predicted_outcome': row[2],
                'actual_outcome': row[3],
                'accuracy': row[4],
                'timestamp': row[5],
                'status': row[6]
            })
        
        conn.close()
        return timeline
    
    def _get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        # Get all component statistics
        pred_stats = self.predictor.get_prediction_statistics()
        hyp_stats = self.hypothesis_generator.get_hypothesis_statistics()
        reflex_stats = self.reflex_anticipator.get_anticipatory_statistics()
        
        # Calculate overall system health
        system_health = self._calculate_system_health(pred_stats, hyp_stats, reflex_stats)
        
        # Get recent activity
        recent_activity = self._get_recent_activity()
        
        return {
            'prediction_system': pred_stats,
            'hypothesis_system': hyp_stats,
            'anticipatory_system': reflex_stats,
            'system_health': system_health,
            'recent_activity': recent_activity
        }
    
    def _calculate_system_health(self, pred_stats: Dict[str, Any], 
                                hyp_stats: Dict[str, Any], 
                                reflex_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall system health metrics."""
        # Prediction accuracy
        pred_accuracy = pred_stats.get('average_accuracy', 0.0)
        
        # Hypothesis confirmation rate
        hyp_confirmation = hyp_stats.get('confirmation_rate', 0.0)
        
        # Reflex success rate
        reflex_success = reflex_stats.get('success_rate', 0.0)
        
        # Overall health score
        health_score = (pred_accuracy + hyp_confirmation + reflex_success) / 3.0
        
        # Determine health status
        if health_score >= 0.8:
            status = 'excellent'
        elif health_score >= 0.6:
            status = 'good'
        elif health_score >= 0.4:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'overall_score': health_score,
            'status': status,
            'prediction_accuracy': pred_accuracy,
            'hypothesis_confirmation_rate': hyp_confirmation,
            'reflex_success_rate': reflex_success
        }
    
    def _get_recent_activity(self) -> Dict[str, Any]:
        """Get recent system activity."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Recent predictions (last 24 hours)
        cursor.execute("""
            SELECT COUNT(*) FROM predictive_goals 
            WHERE created_at > ?
        """, (time.time() - 24 * 3600,))
        recent_predictions = cursor.fetchone()[0]
        
        # Recent hypotheses (last 24 hours)
        cursor.execute("""
            SELECT COUNT(*) FROM hypotheses 
            WHERE created_at > ?
        """, (time.time() - 24 * 3600,))
        recent_hypotheses = cursor.fetchone()[0]
        
        # Recent reflexes (last 24 hours)
        cursor.execute("""
            SELECT COUNT(*) FROM anticipatory_reflexes 
            WHERE created_at > ?
        """, (time.time() - 24 * 3600,))
        recent_reflexes = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'predictions_24h': recent_predictions,
            'hypotheses_24h': recent_hypotheses,
            'reflexes_24h': recent_reflexes,
            'total_activity_24h': recent_predictions + recent_hypotheses + recent_reflexes
        }
    
    def _store_dashboard_metrics(self, metrics: DashboardMetrics):
        """Store dashboard metrics for history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO dashboard_metrics 
            (timestamp, total_predictions, pending_predictions, total_hypotheses,
             open_hypotheses, total_anticipatory_reflexes, successful_reflexes,
             prediction_accuracy, hypothesis_confirmation_rate, reflex_success_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.timestamp, metrics.total_predictions, metrics.pending_predictions,
            metrics.total_hypotheses, metrics.open_hypotheses, metrics.total_anticipatory_reflexes,
            metrics.successful_reflexes, metrics.prediction_accuracy,
            metrics.hypothesis_confirmation_rate, metrics.reflex_success_rate
        ))
        
        conn.commit()
        conn.close()
        
        # Clean up old metrics
        self._cleanup_old_metrics()
    
    def _cleanup_old_metrics(self):
        """Clean up old dashboard metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Keep only the most recent metrics
        cursor.execute("""
            DELETE FROM dashboard_metrics 
            WHERE timestamp NOT IN (
                SELECT timestamp FROM dashboard_metrics 
                ORDER BY timestamp DESC 
                LIMIT ?
            )
        """, (self.metrics_history_size,))
        
        conn.commit()
        conn.close()
    
    def record_outcome(self, prediction_id: str, predicted_outcome: str, 
                      actual_outcome: Optional[str] = None, accuracy: Optional[float] = None):
        """Record a prediction outcome."""
        timeline_id = f"timeline_{prediction_id}_{int(time.time())}"
        
        status = 'confirmed' if actual_outcome else 'pending'
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO outcome_timeline 
            (timeline_id, prediction_id, predicted_outcome, actual_outcome,
             accuracy, timestamp, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            timeline_id, prediction_id, predicted_outcome, actual_outcome,
            accuracy, time.time(), status
        ))
        
        conn.commit()
        conn.close()
    
    def get_metrics_history(self, hours: int = 24) -> List[DashboardMetrics]:
        """Get historical metrics for the specified time period."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = time.time() - (hours * 3600)
        
        cursor.execute("""
            SELECT timestamp, total_predictions, pending_predictions, total_hypotheses,
                   open_hypotheses, total_anticipatory_reflexes, successful_reflexes,
                   prediction_accuracy, hypothesis_confirmation_rate, reflex_success_rate
            FROM dashboard_metrics 
            WHERE timestamp > ?
            ORDER BY timestamp ASC
        """, (cutoff_time,))
        
        metrics_history = []
        for row in cursor.fetchall():
            metrics = DashboardMetrics(
                timestamp=row[0],
                total_predictions=row[1],
                pending_predictions=row[2],
                total_hypotheses=row[3],
                open_hypotheses=row[4],
                total_anticipatory_reflexes=row[5],
                successful_reflexes=row[6],
                prediction_accuracy=row[7],
                hypothesis_confirmation_rate=row[8],
                reflex_success_rate=row[9]
            )
            metrics_history.append(metrics)
        
        conn.close()
        return metrics_history
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends over time."""
        metrics_history = self.get_metrics_history(hours)
        
        if not metrics_history:
            return {}
        
        # Extract trends
        timestamps = [m.timestamp for m in metrics_history]
        prediction_accuracies = [m.prediction_accuracy for m in metrics_history]
        hypothesis_rates = [m.hypothesis_confirmation_rate for m in metrics_history]
        reflex_rates = [m.reflex_success_rate for m in metrics_history]
        
        # Calculate trends (simple linear regression)
        def calculate_trend(values):
            if len(values) < 2:
                return 0.0
            x = np.arange(len(values))
            y = np.array(values)
            slope = (len(values) * np.sum(x * y) - np.sum(x) * np.sum(y)) / (len(values) * np.sum(x * x) - np.sum(x) ** 2)
            return slope
        
        trends = {
            'prediction_accuracy_trend': calculate_trend(prediction_accuracies),
            'hypothesis_confirmation_trend': calculate_trend(hypothesis_rates),
            'reflex_success_trend': calculate_trend(reflex_rates),
            'overall_trend': calculate_trend([(a + h + r) / 3 for a, h, r in zip(prediction_accuracies, hypothesis_rates, reflex_rates)])
        }
        
        return trends
    
    def export_dashboard_report(self, format: str = 'json') -> str:
        """Export dashboard data as a report."""
        dashboard_data = self.generate_dashboard_data()
        
        if format == 'json':
            return json.dumps(dashboard_data, indent=2, default=str)
        elif format == 'text':
            return self._format_text_report(dashboard_data)
        else:
            return json.dumps(dashboard_data, indent=2, default=str)
    
    def _format_text_report(self, dashboard_data: Dict[str, Any]) -> str:
        """Format dashboard data as a text report."""
        report = []
        report.append("=" * 60)
        report.append("CAUSAL AUDIT DASHBOARD REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.fromtimestamp(dashboard_data['timestamp'])}")
        report.append("")
        
        # Current metrics
        metrics = dashboard_data['current_metrics']
        report.append("CURRENT METRICS:")
        report.append(f"  Total Predictions: {metrics.total_predictions}")
        report.append(f"  Pending Predictions: {metrics.pending_predictions}")
        report.append(f"  Total Hypotheses: {metrics.total_hypotheses}")
        report.append(f"  Open Hypotheses: {metrics.open_hypotheses}")
        report.append(f"  Anticipatory Reflexes: {metrics.total_anticipatory_reflexes}")
        report.append(f"  Successful Reflexes: {metrics.successful_reflexes}")
        report.append(f"  Prediction Accuracy: {metrics.prediction_accuracy:.2%}")
        report.append(f"  Hypothesis Confirmation Rate: {metrics.hypothesis_confirmation_rate:.2%}")
        report.append(f"  Reflex Success Rate: {metrics.reflex_success_rate:.2%}")
        report.append("")
        
        # Upcoming drifts
        report.append("UPCOMING PREDICTED DRIFTS:")
        for drift in dashboard_data['upcoming_drifts'][:5]:
            report.append(f"  {drift['prediction_type']}: {drift['predicted_outcome']}")
            report.append(f"    Probability: {drift['probability']:.2%}, Urgency: {drift['urgency']:.2f}")
        report.append("")
        
        # System health
        health = dashboard_data['system_statistics']['system_health']
        report.append(f"SYSTEM HEALTH: {health['status'].upper()} ({health['overall_score']:.2%})")
        report.append("")
        
        return "\n".join(report) 