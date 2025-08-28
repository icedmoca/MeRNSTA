#!/usr/bin/env python3
"""
Causal Drift Predictor for MeRNSTA

This module implements predictive modeling to forecast likely future drift patterns
and trigger preemptive repairs before symptoms appear.
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
import hashlib

from .enhanced_memory_model import EnhancedTripletFact


@dataclass
class TimeSeriesPoint:
    """Represents a single time-series data point for a token."""
    timestamp: float
    token_id: int
    volatility: float
    drift_score: float
    coherence: float
    consistency_rate: float
    contradiction_count: int = 0
    fact_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeSeriesPoint':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PredictiveGoal:
    """Represents a predictive goal spawned from drift prediction."""
    goal_id: str
    token_id: int
    cluster_id: Optional[str]
    prediction_type: str  # 'contradiction_prevention', 'coherence_improvement', 'volatility_reduction'
    predicted_outcome: str
    source_metrics: Dict[str, float]
    probability: float
    urgency: float  # 0.0 to 1.0
    created_at: float = 0.0
    executed_at: Optional[float] = None
    reflex_cycle_id: Optional[str] = None
    status: str = 'pending'  # pending, executing, completed, failed
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictiveGoal':
        """Create from dictionary."""
        return cls(**data)


class CausalDriftPredictor:
    """
    Causal Drift Predictor that forecasts likely future drift patterns.
    
    Features:
    - Maintains time-series data for each token
    - Fits predictive models to forecast drift
    - Spawns predictive goals when thresholds are exceeded
    - Links predictions to reflex cycles for tracking
    """
    
    def __init__(self, db_path: str = "causal_predictions.db"):
        self.db_path = db_path
        
        # Prediction parameters
        self.prediction_window = 5  # Predict next 5 tasks
        self.volatility_threshold = 0.7  # High volatility threshold
        self.contradiction_probability_threshold = 0.6  # Likelihood threshold
        self.coherence_decay_threshold = 0.3  # Coherence decay threshold
        
        # Model parameters
        self.min_data_points = 10  # Minimum points for prediction
        self.rolling_window = 20  # Rolling window for trend analysis
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize causal predictions database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Time series data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS time_series_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                token_id INTEGER NOT NULL,
                volatility REAL NOT NULL,
                drift_score REAL NOT NULL,
                coherence REAL NOT NULL,
                consistency_rate REAL NOT NULL,
                contradiction_count INTEGER DEFAULT 0,
                fact_count INTEGER DEFAULT 0,
                INDEX(token_id, timestamp)
            )
        """)
        
        # Predictive goals
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictive_goals (
                goal_id TEXT PRIMARY KEY,
                token_id INTEGER NOT NULL,
                cluster_id TEXT,
                prediction_type TEXT NOT NULL,
                predicted_outcome TEXT NOT NULL,
                source_metrics TEXT NOT NULL,
                probability REAL NOT NULL,
                urgency REAL NOT NULL,
                created_at REAL NOT NULL,
                executed_at REAL,
                reflex_cycle_id TEXT,
                status TEXT DEFAULT 'pending',
                INDEX(token_id),
                INDEX(status),
                INDEX(created_at)
            )
        """)
        
        # Prediction history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                prediction_id TEXT PRIMARY KEY,
                goal_id TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                predicted_value REAL NOT NULL,
                actual_value REAL,
                accuracy REAL,
                timestamp REAL NOT NULL,
                status TEXT DEFAULT 'pending',  # pending, confirmed, failed
                INDEX(goal_id),
                INDEX(status),
                INDEX(timestamp)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_token_metrics(self, token_id: int, metrics: Dict[str, float]):
        """
        Record current metrics for a token.
        
        Args:
            token_id: The token identifier
            metrics: Dictionary containing volatility, drift_score, coherence, consistency_rate
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO time_series_data 
            (timestamp, token_id, volatility, drift_score, coherence, consistency_rate, 
             contradiction_count, fact_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(), token_id,
            metrics.get('volatility', 0.0),
            metrics.get('drift_score', 0.0),
            metrics.get('coherence', 0.0),
            metrics.get('consistency_rate', 0.0),
            metrics.get('contradiction_count', 0),
            metrics.get('fact_count', 0)
        ))
        
        conn.commit()
        conn.close()
        
        # Check if we should make predictions
        self._check_and_predict(token_id)
    
    def _check_and_predict(self, token_id: int):
        """Check if we have enough data and make predictions."""
        time_series = self._get_time_series(token_id)
        
        if len(time_series) < self.min_data_points:
            return
        
        # Make predictions
        predictions = self._make_predictions(token_id, time_series)
        
        # Check if any predictions exceed thresholds
        for pred_type, prediction in predictions.items():
            if self._should_spawn_goal(pred_type, prediction):
                self._spawn_predictive_goal(token_id, pred_type, prediction)
    
    def _get_time_series(self, token_id: int) -> List[TimeSeriesPoint]:
        """Get time series data for a token."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, token_id, volatility, drift_score, coherence, 
                   consistency_rate, contradiction_count, fact_count
            FROM time_series_data 
            WHERE token_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (token_id, self.rolling_window))
        
        time_series = []
        for row in cursor.fetchall():
            point = TimeSeriesPoint(
                timestamp=row[0],
                token_id=row[1],
                volatility=row[2],
                drift_score=row[3],
                coherence=row[4],
                consistency_rate=row[5],
                contradiction_count=row[6],
                fact_count=row[7]
            )
            time_series.append(point)
        
        conn.close()
        return list(reversed(time_series))  # Return in chronological order
    
    def _make_predictions(self, token_id: int, time_series: List[TimeSeriesPoint]) -> Dict[str, Dict[str, Any]]:
        """Make predictions for various metrics."""
        predictions = {}
        
        if len(time_series) < 3:
            return predictions
        
        # Extract recent values
        recent_volatility = [p.volatility for p in time_series[-5:]]
        recent_drift = [p.drift_score for p in time_series[-5:]]
        recent_coherence = [p.coherence for p in time_series[-5:]]
        recent_consistency = [p.consistency_rate for p in time_series[-5:]]
        
        # Predict volatility trend
        if len(recent_volatility) >= 3:
            volatility_trend = self._calculate_trend(recent_volatility)
            predicted_volatility = recent_volatility[-1] + (volatility_trend * self.prediction_window)
            
            predictions['volatility'] = {
                'current': recent_volatility[-1],
                'predicted': predicted_volatility,
                'trend': volatility_trend,
                'probability': self._calculate_probability(volatility_trend, recent_volatility)
            }
        
        # Predict contradiction likelihood
        contradiction_prob = self._predict_contradiction_likelihood(time_series)
        if contradiction_prob > 0:
            predictions['contradiction'] = {
                'current_probability': contradiction_prob,
                'predicted_probability': min(1.0, contradiction_prob * 1.2),  # Assume increasing
                'trend': 'increasing' if contradiction_prob > 0.3 else 'stable',
                'probability': contradiction_prob
            }
        
        # Predict coherence decay
        if len(recent_coherence) >= 3:
            coherence_trend = self._calculate_trend(recent_coherence)
            predicted_coherence = recent_coherence[-1] + (coherence_trend * self.prediction_window)
            
            predictions['coherence'] = {
                'current': recent_coherence[-1],
                'predicted': predicted_coherence,
                'trend': coherence_trend,
                'probability': self._calculate_probability(coherence_trend, recent_coherence)
            }
        
        return predictions
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using simple linear regression."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        # Simple linear regression
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x * x) - np.sum(x) ** 2)
        return slope
    
    def _calculate_probability(self, trend: float, values: List[float]) -> float:
        """Calculate probability based on trend and variance."""
        if not values:
            return 0.0
        
        # Normalize trend to 0-1 range
        trend_prob = min(1.0, max(0.0, (trend + 0.1) / 0.2))
        
        # Consider variance
        variance = np.var(values) if len(values) > 1 else 0.0
        variance_factor = min(1.0, variance * 10)  # Higher variance = higher uncertainty
        
        return trend_prob * (1.0 - variance_factor * 0.3)
    
    def _predict_contradiction_likelihood(self, time_series: List[TimeSeriesPoint]) -> float:
        """Predict likelihood of contradiction emerging."""
        if len(time_series) < 3:
            return 0.0
        
        # Factors that increase contradiction likelihood
        recent_points = time_series[-3:]
        
        # High volatility
        avg_volatility = np.mean([p.volatility for p in recent_points])
        volatility_factor = min(1.0, avg_volatility / 0.5)
        
        # Low coherence
        avg_coherence = np.mean([p.coherence for p in recent_points])
        coherence_factor = 1.0 - avg_coherence
        
        # Recent contradictions
        recent_contradictions = sum(p.contradiction_count for p in recent_points)
        contradiction_factor = min(1.0, recent_contradictions / 3.0)
        
        # Combine factors
        likelihood = (volatility_factor * 0.4 + coherence_factor * 0.4 + contradiction_factor * 0.2)
        return min(1.0, likelihood)
    
    def _should_spawn_goal(self, pred_type: str, prediction: Dict[str, Any]) -> bool:
        """Check if prediction exceeds thresholds and should spawn a goal."""
        if pred_type == 'volatility':
            return prediction['predicted'] > self.volatility_threshold
        elif pred_type == 'contradiction':
            return prediction['predicted_probability'] > self.contradiction_probability_threshold
        elif pred_type == 'coherence':
            return prediction['predicted'] < self.coherence_decay_threshold
        return False
    
    def _spawn_predictive_goal(self, token_id: int, pred_type: str, prediction: Dict[str, Any]):
        """Spawn a predictive goal based on prediction."""
        goal_id = f"predictive_{pred_type}_{token_id}_{int(time.time())}"
        
        # Determine prediction type and outcome
        if pred_type == 'volatility':
            prediction_type = 'volatility_reduction'
            predicted_outcome = f"Volatility will increase to {prediction['predicted']:.2f}"
        elif pred_type == 'contradiction':
            prediction_type = 'contradiction_prevention'
            predicted_outcome = f"Contradiction likelihood will reach {prediction['predicted_probability']:.2f}"
        elif pred_type == 'coherence':
            prediction_type = 'coherence_improvement'
            predicted_outcome = f"Coherence will decay to {prediction['predicted']:.2f}"
        else:
            return
        
        # Calculate urgency based on probability and trend
        urgency = prediction.get('probability', 0.5)
        if prediction.get('trend', 0) > 0:
            urgency *= 1.2  # Increasing trend increases urgency
        
        # Create predictive goal
        goal = PredictiveGoal(
            goal_id=goal_id,
            token_id=token_id,
            cluster_id=self._get_cluster_id(token_id),
            prediction_type=prediction_type,
            predicted_outcome=predicted_outcome,
            source_metrics=prediction,
            probability=prediction.get('probability', 0.5),
            urgency=min(1.0, urgency)
        )
        
        # Store goal
        self._store_predictive_goal(goal)
        
        logging.info(f"[CausalDriftPredictor] Spawned predictive goal {goal_id} for {pred_type}")
    
    def _get_cluster_id(self, token_id: int) -> Optional[str]:
        """Get cluster ID for a token (placeholder implementation)."""
        # This would need to be implemented based on how clusters are stored
        return f"cluster_{token_id % 10}"  # Simple placeholder
    
    def _store_predictive_goal(self, goal: PredictiveGoal):
        """Store predictive goal in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictive_goals 
            (goal_id, token_id, cluster_id, prediction_type, predicted_outcome,
             source_metrics, probability, urgency, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            goal.goal_id, goal.token_id, goal.cluster_id, goal.prediction_type,
            goal.predicted_outcome, json.dumps(goal.source_metrics),
            goal.probability, goal.urgency, goal.created_at, goal.status
        ))
        
        conn.commit()
        conn.close()
    
    def get_pending_predictive_goals(self, limit: Optional[int] = None) -> List[PredictiveGoal]:
        """Get pending predictive goals."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT goal_id, token_id, cluster_id, prediction_type, predicted_outcome,
                   source_metrics, probability, urgency, created_at, executed_at,
                   reflex_cycle_id, status
            FROM predictive_goals 
            WHERE status = 'pending'
            ORDER BY urgency DESC, created_at ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        
        goals = []
        for row in cursor.fetchall():
            data = {
                'goal_id': row[0],
                'token_id': row[1],
                'cluster_id': row[2],
                'prediction_type': row[3],
                'predicted_outcome': row[4],
                'source_metrics': json.loads(row[5]),
                'probability': row[6],
                'urgency': row[7],
                'created_at': row[8],
                'executed_at': row[9],
                'reflex_cycle_id': row[10],
                'status': row[11]
            }
            goals.append(PredictiveGoal.from_dict(data))
        
        conn.close()
        return goals
    
    def mark_goal_executed(self, goal_id: str, reflex_cycle_id: str):
        """Mark a predictive goal as executed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE predictive_goals 
            SET executed_at = ?, reflex_cycle_id = ?, status = 'executing'
            WHERE goal_id = ?
        """, (time.time(), reflex_cycle_id, goal_id))
        
        conn.commit()
        conn.close()
    
    def mark_goal_completed(self, goal_id: str, success: bool = True):
        """Mark a predictive goal as completed."""
        status = 'completed' if success else 'failed'
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE predictive_goals 
            SET status = ?
            WHERE goal_id = ?
        """, (status, goal_id))
        
        conn.commit()
        conn.close()
    
    def record_prediction_outcome(self, goal_id: str, actual_value: float, accuracy: float):
        """Record the outcome of a prediction."""
        prediction_id = f"pred_{goal_id}_{int(time.time())}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO prediction_history 
            (prediction_id, goal_id, prediction_type, predicted_value, actual_value,
             accuracy, timestamp, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction_id, goal_id, 'drift_prediction', 0.0, actual_value,
            accuracy, time.time(), 'confirmed'
        ))
        
        conn.commit()
        conn.close()
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get statistics about predictions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM predictive_goals")
        total_predictions = cursor.fetchone()[0]
        
        # Pending predictions
        cursor.execute("SELECT COUNT(*) FROM predictive_goals WHERE status = 'pending'")
        pending_predictions = cursor.fetchone()[0]
        
        # Completed predictions
        cursor.execute("SELECT COUNT(*) FROM predictive_goals WHERE status = 'completed'")
        completed_predictions = cursor.fetchone()[0]
        
        # Average accuracy
        cursor.execute("SELECT AVG(accuracy) FROM prediction_history")
        avg_accuracy = cursor.fetchone()[0] or 0.0
        
        # Recent predictions (last 24 hours)
        cursor.execute("""
            SELECT COUNT(*) FROM predictive_goals 
            WHERE created_at > ?
        """, (time.time() - 24 * 3600,))
        recent_predictions = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_predictions': total_predictions,
            'pending_predictions': pending_predictions,
            'completed_predictions': completed_predictions,
            'average_accuracy': avg_accuracy,
            'recent_predictions_24h': recent_predictions
        }
    
    def get_token_predictions(self, token_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent predictions for a specific token."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT goal_id, prediction_type, predicted_outcome, probability, urgency,
                   created_at, status
            FROM predictive_goals 
            WHERE token_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (token_id, limit))
        
        predictions = []
        for row in cursor.fetchall():
            predictions.append({
                'goal_id': row[0],
                'prediction_type': row[1],
                'predicted_outcome': row[2],
                'probability': row[3],
                'urgency': row[4],
                'created_at': row[5],
                'status': row[6]
            })
        
        conn.close()
        return predictions 