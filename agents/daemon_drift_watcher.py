#!/usr/bin/env python3
"""
DaemonDriftWatcher - Background drift prediction daemon

Monitors token history for early signs of cognitive decay and spawns
anticipatory meta-goals to preempt drift before it becomes problematic.

Features:
- Token volatility trend analysis
- Contradiction frequency monitoring  
- Cluster entropy change detection
- Anticipatory goal generation
- Background monitoring with configurable intervals
"""

import time
import logging
import threading
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from agents.cognitive_repair_agent import DriftTriggeredGoal, get_cognitive_repair_agent
from storage.enhanced_memory_system import EnhancedMemorySystem
from storage.token_graph import get_token_graph
from storage.reflex_log import get_reflex_logger
from config.settings import get_config


@dataclass
class DriftPrediction:
    """Represents a drift prediction for a token."""
    token_id: int
    prediction_type: str  # volatility, contradiction, entropy
    confidence: float  # 0.0 to 1.0
    trend_direction: str  # increasing, decreasing, stable
    current_value: float
    threshold_value: float
    justification: str
    timestamp: float
    cluster_id: Optional[str] = None


@dataclass
class AnticipatoryGoal:
    """Represents an anticipatory goal spawned by the drift watcher."""
    goal_id: str
    token_id: int
    prediction: DriftPrediction
    goal_text: str
    priority: float
    created_time: float
    status: str = "pending"  # pending, executing, completed, failed
    cluster_id: Optional[str] = None


class DaemonDriftWatcher:
    """
    Background daemon that monitors for drift patterns and spawns anticipatory goals.
    
    Features:
    - Periodic token volatility analysis
    - Contradiction frequency monitoring
    - Cluster entropy change detection
    - Anticipatory goal generation
    - Background monitoring with configurable intervals
    """
    
    def __init__(self, memory_system: EnhancedMemorySystem = None):
        self.memory_system = memory_system or EnhancedMemorySystem()
        self.repair_agent = get_cognitive_repair_agent()
        self.token_graph = get_token_graph()
        self.reflex_logger = get_reflex_logger()
        
        # Configuration
        config = get_config()
        self.monitoring_interval = config.get('drift_watcher_interval', 300)  # 5 minutes
        self.volatility_threshold = config.get('drift_volatility_threshold', 0.7)
        self.contradiction_threshold = config.get('drift_contradiction_threshold', 0.3)
        self.entropy_threshold = config.get('drift_entropy_threshold', 0.8)
        self.prediction_confidence_threshold = config.get('drift_prediction_confidence', 0.6)
        
        # Monitoring state
        self.running = False
        self.monitoring_thread = None
        self.last_check_time = time.time()
        self.predicted_tokens: Set[int] = set()
        self.anticipatory_goals: Dict[str, AnticipatoryGoal] = {}
        
        # Performance tracking
        self.prediction_count = 0
        self.goal_spawn_count = 0
        self.successful_predictions = 0
        
        print(f"[DaemonDriftWatcher] Initialized with interval={self.monitoring_interval}s, "
              f"volatility_threshold={self.volatility_threshold}, "
              f"contradiction_threshold={self.contradiction_threshold}")
    
    def start_monitoring(self):
        """Start the background monitoring daemon."""
        if self.running:
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print(f"[DaemonDriftWatcher] Started background monitoring daemon")
    
    def stop_monitoring(self):
        """Stop the background monitoring daemon."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print(f"[DaemonDriftWatcher] Stopped background monitoring daemon")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                self._check_for_drift_patterns()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logging.error(f"[DaemonDriftWatcher] Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _check_for_drift_patterns(self):
        """Check for drift patterns and spawn anticipatory goals."""
        try:
            print(f"[DaemonDriftWatcher] Checking for drift patterns...")
            
            # Get all tokens to monitor
            tokens_to_monitor = self._get_tokens_to_monitor()
            
            predictions = []
            
            for token_id in tokens_to_monitor:
                # Check volatility trends
                volatility_prediction = self._check_volatility_trend(token_id)
                if volatility_prediction:
                    predictions.append(volatility_prediction)
                
                # Check contradiction frequency
                contradiction_prediction = self._check_contradiction_frequency(token_id)
                if contradiction_prediction:
                    predictions.append(contradiction_prediction)
                
                # Check cluster entropy (if available)
                entropy_prediction = self._check_cluster_entropy(token_id)
                if entropy_prediction:
                    predictions.append(entropy_prediction)
            
            # Spawn anticipatory goals for high-confidence predictions
            for prediction in predictions:
                if prediction.confidence >= self.prediction_confidence_threshold:
                    self._spawn_anticipatory_goal(prediction)
            
            self.last_check_time = time.time()
            print(f"[DaemonDriftWatcher] Found {len(predictions)} drift predictions, "
                  f"spawned {len([p for p in predictions if p.confidence >= self.prediction_confidence_threshold])} goals")
            
        except Exception as e:
            logging.error(f"[DaemonDriftWatcher] Error checking drift patterns: {e}")
    
    def _get_tokens_to_monitor(self) -> List[int]:
        """Get list of tokens to monitor for drift patterns."""
        try:
            # Get tokens from memory system
            facts = self.memory_system.get_facts(limit=1000)
            token_ids = set()
            
            for fact in facts:
                if hasattr(fact, 'token_ids') and fact.token_ids:
                    token_ids.update(fact.token_ids)
            
            # Add tokens from token graph if available
            if self.token_graph and hasattr(self.token_graph, 'graph'):
                token_ids.update(self.token_graph.graph.keys())
            
            return list(token_ids)[:100]  # Limit to 100 tokens for performance
            
        except Exception as e:
            logging.error(f"[DaemonDriftWatcher] Error getting tokens to monitor: {e}")
            return []
    
    def _check_volatility_trend(self, token_id: int) -> Optional[DriftPrediction]:
        """Check volatility trend for a token."""
        try:
            # Get recent reflex scores for this token
            recent_scores = self.reflex_logger.get_scores_by_token(token_id, limit=10)
            
            if len(recent_scores) < 3:
                return None
            
            # Calculate volatility trend
            recent_volatility = [score.volatility_delta for score in recent_scores[:5]]
            avg_volatility = sum(recent_volatility) / len(recent_volatility)
            
            # Check if volatility is increasing
            if avg_volatility > self.volatility_threshold:
                confidence = min(avg_volatility, 1.0)
                trend_direction = "increasing" if avg_volatility > 0 else "stable"
                
                return DriftPrediction(
                    token_id=token_id,
                    prediction_type="volatility",
                    confidence=confidence,
                    trend_direction=trend_direction,
                    current_value=avg_volatility,
                    threshold_value=self.volatility_threshold,
                    justification=f"Token {token_id} showing increasing volatility trend (avg: {avg_volatility:.3f})",
                    timestamp=time.time()
                )
            
            return None
            
        except Exception as e:
            logging.error(f"[DaemonDriftWatcher] Error checking volatility trend: {e}")
            return None
    
    def _check_contradiction_frequency(self, token_id: int) -> Optional[DriftPrediction]:
        """Check contradiction frequency for a token."""
        try:
            # Get facts related to this token
            facts = self.memory_system.get_facts(limit=100)
            token_facts = [f for f in facts if hasattr(f, 'token_ids') and token_id in f.token_ids]
            
            if len(token_facts) < 3:
                return None
            
            # Calculate contradiction rate
            contradictions = sum(1 for f in token_facts if hasattr(f, 'contradiction') and f.contradiction)
            contradiction_rate = contradictions / len(token_facts)
            
            # Check if contradiction rate is high
            if contradiction_rate > self.contradiction_threshold:
                confidence = min(contradiction_rate, 1.0)
                trend_direction = "increasing" if contradiction_rate > 0.5 else "stable"
                
                return DriftPrediction(
                    token_id=token_id,
                    prediction_type="contradiction",
                    confidence=confidence,
                    trend_direction=trend_direction,
                    current_value=contradiction_rate,
                    threshold_value=self.contradiction_threshold,
                    justification=f"Token {token_id} has high contradiction rate ({contradiction_rate:.3f})",
                    timestamp=time.time()
                )
            
            return None
            
        except Exception as e:
            logging.error(f"[DaemonDriftWatcher] Error checking contradiction frequency: {e}")
            return None
    
    def _check_cluster_entropy(self, token_id: int) -> Optional[DriftPrediction]:
        """Check cluster entropy for a token."""
        try:
            if not self.token_graph:
                return None
            
            # Get cluster for this token
            cluster = self.token_graph.get_cluster_by_token(token_id)
            if not cluster:
                return None
            
            # Check cluster entropy if available
            if hasattr(cluster, 'entropy_score'):
                entropy = cluster.entropy_score
                
                if entropy > self.entropy_threshold:
                    confidence = min(entropy, 1.0)
                    trend_direction = "increasing" if entropy > 0.8 else "stable"
                    
                    return DriftPrediction(
                        token_id=token_id,
                        prediction_type="entropy",
                        confidence=confidence,
                        trend_direction=trend_direction,
                        current_value=entropy,
                        threshold_value=self.entropy_threshold,
                        justification=f"Token {token_id} cluster has high entropy ({entropy:.3f})",
                        timestamp=time.time(),
                        cluster_id=cluster.cluster_id
                    )
            
            return None
            
        except Exception as e:
            logging.error(f"[DaemonDriftWatcher] Error checking cluster entropy: {e}")
            return None
    
    def _spawn_anticipatory_goal(self, prediction: DriftPrediction):
        """Spawn an anticipatory goal based on a drift prediction."""
        try:
            goal_id = f"anticipate_{prediction.prediction_type}_{prediction.token_id}_{int(time.time())}"
            
            # Create goal text based on prediction type
            if prediction.prediction_type == "volatility":
                goal_text = f"anticipate drift on token {prediction.token_id} showing volatility instability"
            elif prediction.prediction_type == "contradiction":
                goal_text = f"anticipate drift on token {prediction.token_id} showing contradiction instability"
            elif prediction.prediction_type == "entropy":
                goal_text = f"anticipate drift on token {prediction.token_id} showing entropy instability"
            else:
                goal_text = f"anticipate drift on token {prediction.token_id} showing instability"
            
            # Create anticipatory goal
            anticipatory_goal = AnticipatoryGoal(
                goal_id=goal_id,
                token_id=prediction.token_id,
                prediction=prediction,
                goal_text=goal_text,
                priority=prediction.confidence,
                created_time=time.time(),
                cluster_id=prediction.cluster_id
            )
            
            # Create drift-triggered goal for the repair agent
            drift_goal = DriftTriggeredGoal(
                goal_id=goal_id,
                token_id=prediction.token_id,
                drift_score=prediction.confidence,
                goal=goal_text,
                priority=prediction.confidence,
                repair_strategy="anticipatory_drift",
                affected_facts=[],
                tags=["#anticipation"]
            )
            
            # Add to repair agent
            self.repair_agent.add_drift_goal(drift_goal)
            
            # Store anticipatory goal
            self.anticipatory_goals[goal_id] = anticipatory_goal
            self.predicted_tokens.add(prediction.token_id)
            
            # Log to memory
            self._log_anticipatory_goal(anticipatory_goal)
            
            self.goal_spawn_count += 1
            print(f"[DaemonDriftWatcher] Spawned anticipatory goal: {goal_id} "
                  f"(confidence: {prediction.confidence:.3f})")
            
        except Exception as e:
            logging.error(f"[DaemonDriftWatcher] Error spawning anticipatory goal: {e}")
    
    def _log_anticipatory_goal(self, goal: AnticipatoryGoal):
        """Log anticipatory goal to memory system."""
        try:
            from storage.enhanced_memory_model import EnhancedTripletFact
            
            # Create memory fact for the anticipatory goal
            fact = EnhancedTripletFact(
                subject="system",
                predicate="spawned_anticipatory_goal",
                object=goal.goal_text,
                confidence=goal.priority,
                source="DaemonDriftWatcher",
                metadata={
                    "goal_id": goal.goal_id,
                    "token_id": goal.token_id,
                    "prediction_type": goal.prediction.prediction_type,
                    "confidence": goal.prediction.confidence,
                    "anticipatory": True,
                    "created_time": goal.created_time
                }
            )
            
            self.memory_system.store_fact(fact)
            
        except Exception as e:
            logging.error(f"[DaemonDriftWatcher] Error logging anticipatory goal: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring status and statistics."""
        return {
            "running": self.running,
            "last_check_time": self.last_check_time,
            "prediction_count": self.prediction_count,
            "goal_spawn_count": self.goal_spawn_count,
            "successful_predictions": self.successful_predictions,
            "predicted_tokens_count": len(self.predicted_tokens),
            "active_anticipatory_goals": len(self.anticipatory_goals),
            "monitoring_interval": self.monitoring_interval
        }
    
    def get_predicted_tokens(self) -> List[Dict[str, Any]]:
        """Get list of currently predicted tokens."""
        tokens = []
        for goal in self.anticipatory_goals.values():
            tokens.append({
                "token_id": goal.token_id,
                "prediction_type": goal.prediction.prediction_type,
                "confidence": goal.prediction.confidence,
                "goal_id": goal.goal_id,
                "status": goal.status,
                "created_time": goal.created_time
            })
        return tokens


# Global daemon instance
_daemon_instance = None


def get_daemon_drift_watcher(memory_system: EnhancedMemorySystem = None) -> DaemonDriftWatcher:
    """Get or create the global daemon drift watcher instance."""
    global _daemon_instance
    
    if _daemon_instance is None:
        _daemon_instance = DaemonDriftWatcher(memory_system)
    
    return _daemon_instance


def start_drift_watcher(memory_system: EnhancedMemorySystem = None):
    """Start the drift watcher daemon."""
    daemon = get_daemon_drift_watcher(memory_system)
    daemon.start_monitoring()


def stop_drift_watcher():
    """Stop the drift watcher daemon."""
    global _daemon_instance
    if _daemon_instance:
        _daemon_instance.stop_monitoring()


def get_drift_watcher_status() -> Dict[str, Any]:
    """Get drift watcher status."""
    global _daemon_instance
    if _daemon_instance:
        return _daemon_instance.get_monitoring_status()
    return {"running": False} 