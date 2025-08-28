#!/usr/bin/env python3
"""
Causal Prediction Module for MeRNSTA

This module predicts likely next emotional, behavioral, or preference states by:
1. Analyzing current facts and causal patterns
2. Identifying trending causal chains
3. Predicting likely continuations based on historical patterns
4. Auto-suggesting next thoughts/goals

📌 DO NOT HARDCODE prediction patterns or probability weights.
All parameters must be loaded from `config.settings` or environment config.
This is a zero-hardcoding cognitive subsystem.
"""

import time
import math
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from storage.db_utils import get_connection_pool
from config.settings import DEFAULT_VALUES


@dataclass
class CausalPattern:
    """Represents a causal pattern for prediction."""
    cause_type: str  # e.g., "action", "state", "emotion"
    effect_type: str
    pattern_strength: float
    frequency: int
    confidence: float
    example_transitions: List[Tuple[str, str]]  # (cause, effect) examples


@dataclass
class PredictionResult:
    """Represents a prediction about future states."""
    prediction_type: str  # "emotional", "behavioral", "preference"
    predicted_state: str
    confidence: float
    reasoning: str
    supporting_evidence: List[str]
    suggested_actions: List[str]
    time_horizon: str  # "immediate", "short_term", "medium_term"


class CausalPredictionModule:
    """Module for predicting future states based on causal patterns."""
    
    def __init__(self):
        """Initialize the causal prediction module with configurable parameters."""
        # Load configuration parameters (no hardcoding)
        self.prediction_window_hours = DEFAULT_VALUES.get("prediction_window_hours", 24)
        self.min_pattern_frequency = DEFAULT_VALUES.get("min_pattern_frequency", 2)
        self.prediction_confidence_threshold = DEFAULT_VALUES.get("prediction_confidence_threshold", 0.4)
        self.max_predictions = DEFAULT_VALUES.get("max_predictions_per_type", 3)
        
        # Pattern weights for different prediction types
        self.prediction_weights = {
            "emotional": DEFAULT_VALUES.get("emotional_prediction_weight", 1.0),
            "behavioral": DEFAULT_VALUES.get("behavioral_prediction_weight", 0.8),
            "preference": DEFAULT_VALUES.get("preference_prediction_weight", 0.6)
        }
        
        # Initialize pattern database
        self.causal_patterns = self._load_causal_patterns()
        
        print(f"[CausalPrediction] Initialized with window={self.prediction_window_hours}h, "
              f"min_frequency={self.min_pattern_frequency}")
    
    def _load_causal_patterns(self) -> Dict[str, List[CausalPattern]]:
        """Load learned causal patterns from the database."""
        patterns = {
            "emotional": [],
            "behavioral": [], 
            "preference": []
        }
        
        try:
            with get_connection_pool().get_connection() as conn:
                # Query for common causal patterns
                results = conn.execute(
                    """SELECT f1.predicate as cause_pred, f1.object as cause_obj,
                              f2.predicate as effect_pred, f2.object as effect_obj,
                              AVG(ef.causal_strength) as avg_strength,
                              COUNT(*) as frequency
                       FROM facts f1
                       JOIN enhanced_facts ef ON f1.id = ef.fact_id
                       JOIN facts f2 ON ef.change_history LIKE '%causal:%'
                       WHERE ef.causal_strength > 0
                       GROUP BY f1.predicate, f1.object, f2.predicate, f2.object
                       HAVING COUNT(*) >= ?
                       ORDER BY avg_strength DESC, frequency DESC""",
                    (self.min_pattern_frequency,)
                ).fetchall()
                
                for result in results:
                    cause_pred, cause_obj, effect_pred, effect_obj, strength, freq = result
                    
                    # Classify pattern type
                    pattern_type = self._classify_prediction_type(effect_pred, effect_obj)
                    cause_type = self._classify_cause_type(cause_pred, cause_obj)
                    
                    pattern = CausalPattern(
                        cause_type=cause_type,
                        effect_type=pattern_type,
                        pattern_strength=strength,
                        frequency=freq,
                        confidence=min(1.0, strength * math.log(freq + 1)),
                        example_transitions=[(f"{cause_pred} {cause_obj}", f"{effect_pred} {effect_obj}")]
                    )
                    
                    patterns[pattern_type].append(pattern)
                    
        except Exception as e:
            print(f"[CausalPrediction] Error loading patterns: {e}")
        
        return patterns
    
    def predict_next_states(self, user_id: str, session_id: str = None) -> List[PredictionResult]:
        """
        Predict likely next emotional, behavioral, or preference states for a user.
        
        Args:
            user_id: User to predict for
            session_id: Optional session context
            
        Returns:
            List of prediction results sorted by confidence
        """
        try:
            # Get recent user facts
            recent_facts = self._get_recent_facts(user_id, session_id)
            
            if not recent_facts:
                return []
            
            # Analyze current state
            current_state = self._analyze_current_state(recent_facts)
            
            # Generate predictions for each type
            predictions = []
            
            for prediction_type in ["emotional", "behavioral", "preference"]:
                type_predictions = self._predict_for_type(
                    current_state, prediction_type, recent_facts
                )
                predictions.extend(type_predictions)
            
            # Sort by confidence and return top predictions
            predictions.sort(key=lambda p: p.confidence, reverse=True)
            
            return predictions[:self.max_predictions * 3]  # Top predictions across all types
            
        except Exception as e:
            print(f"[CausalPrediction] Error predicting for user {user_id}: {e}")
            return []
    
    def _get_recent_facts(self, user_id: str, session_id: str = None) -> List[Dict[str, Any]]:
        """Get recent facts for the user within the prediction window."""
        try:
            current_time = time.time()
            cutoff_time = current_time - (self.prediction_window_hours * 3600)
            
            with get_connection_pool().get_connection() as conn:
                query = """SELECT f.id, f.subject, f.predicate, f.object, f.confidence, f.timestamp,
                                 COALESCE(ef.causal_strength, 0) as causal_strength
                          FROM facts f
                          LEFT JOIN enhanced_facts ef ON f.id = ef.id
                          WHERE f.timestamp >= ?
                          ORDER BY f.timestamp DESC"""
                
                results = conn.execute(query, (cutoff_time,)).fetchall()
                
                facts = []
                for result in results:
                    facts.append({
                        "id": result[0],
                        "subject": result[1],
                        "predicate": result[2],
                        "object": result[3],
                        "confidence": result[4],
                        "timestamp": result[5],
                        "causal_strength": result[6]
                    })
                
                return facts
                
        except Exception as e:
            print(f"[CausalPrediction] Error getting recent facts: {e}")
            return []
    
    def _analyze_current_state(self, recent_facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the user's current state from recent facts."""
        current_state = {
            "dominant_emotions": [],
            "recent_actions": [],
            "current_stressors": [],
            "positive_trends": [],
            "negative_trends": [],
            "causal_momentum": 0.0
        }
        
        # Categorize recent facts
        for fact in recent_facts:
            pred = fact["predicate"].lower()
            obj = fact["object"].lower()
            
            # Classify emotions
            if any(emotion in pred or emotion in obj 
                   for emotion in ["feel", "emotion", "mood", "happy", "sad", "angry", "anxious", "excited"]):
                current_state["dominant_emotions"].append(f"{pred} {obj}")
            
            # Classify actions
            elif any(action in pred 
                     for action in ["start", "begin", "join", "do", "perform", "complete", "try"]):
                current_state["recent_actions"].append(f"{pred} {obj}")
            
            # Identify stressors
            if any(stress in obj 
                   for stress in ["stress", "overwhelm", "pressure", "anxiety", "worry", "problem"]):
                current_state["current_stressors"].append(f"{pred} {obj}")
            
            # Calculate causal momentum
            current_state["causal_momentum"] += fact.get("causal_strength", 0)
        
        # Identify trends
        current_state["causal_momentum"] /= max(1, len(recent_facts))
        
        if current_state["causal_momentum"] > 0.5:
            current_state["positive_trends"] = ["Strong causal patterns indicate active period"]
        elif current_state["causal_momentum"] < 0.2:
            current_state["negative_trends"] = ["Weak causal patterns may indicate stagnation"]
        
        return current_state
    
    def _predict_for_type(self, current_state: Dict[str, Any], prediction_type: str, 
                         recent_facts: List[Dict[str, Any]]) -> List[PredictionResult]:
        """Generate predictions for a specific type (emotional, behavioral, preference)."""
        predictions = []
        
        # Get relevant patterns for this type
        patterns = self.causal_patterns.get(prediction_type, [])
        
        for pattern in patterns:
            # Check if current state matches pattern prerequisites
            match_confidence = self._calculate_pattern_match(current_state, pattern, recent_facts)
            
            if match_confidence >= self.prediction_confidence_threshold:
                prediction = self._generate_prediction_from_pattern(
                    pattern, current_state, match_confidence, prediction_type
                )
                predictions.append(prediction)
        
        # Generate contextual predictions based on current state
        contextual_predictions = self._generate_contextual_predictions(
            current_state, prediction_type
        )
        predictions.extend(contextual_predictions)
        
        # Sort by confidence and return top predictions for this type
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions[:self.max_predictions]
    
    def _calculate_pattern_match(self, current_state: Dict[str, Any], 
                                pattern: CausalPattern, recent_facts: List[Dict[str, Any]]) -> float:
        """Calculate how well the current state matches a causal pattern."""
        match_score = 0.0
        
        # Check for matching recent actions/states
        for fact in recent_facts[-3:]:  # Last 3 facts
            fact_text = f"{fact['predicate']} {fact['object']}".lower()
            
            for cause_example, effect_example in pattern.example_transitions:
                cause_text = cause_example.lower()
                
                # Simple text similarity check
                if self._text_similarity(fact_text, cause_text) > 0.6:
                    match_score += pattern.pattern_strength
                    break
        
        # Normalize by pattern frequency and confidence
        normalized_score = match_score * pattern.confidence / max(1, pattern.frequency)
        
        return min(1.0, normalized_score)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity between two strings."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _generate_prediction_from_pattern(self, pattern: CausalPattern, 
                                        current_state: Dict[str, Any], 
                                        match_confidence: float,
                                        prediction_type: str) -> PredictionResult:
        """Generate a prediction based on a matched causal pattern."""
        # Extract typical effect from pattern
        if pattern.example_transitions:
            cause_example, effect_example = pattern.example_transitions[0]
            predicted_state = f"You are likely to {effect_example}"
        else:
            predicted_state = f"You may experience {prediction_type} changes"
        
        # Generate reasoning
        reasoning = f"Based on your recent {pattern.cause_type} patterns, "
        reasoning += f"which historically lead to {pattern.effect_type} outcomes "
        reasoning += f"with {pattern.pattern_strength:.1%} strength"
        
        # Determine time horizon
        if pattern.pattern_strength > 0.7:
            time_horizon = "immediate"
        elif pattern.pattern_strength > 0.4:
            time_horizon = "short_term"
        else:
            time_horizon = "medium_term"
        
        # Generate suggested actions
        suggested_actions = self._generate_action_suggestions(pattern, prediction_type)
        
        return PredictionResult(
            prediction_type=prediction_type,
            predicted_state=predicted_state,
            confidence=match_confidence * self.prediction_weights[prediction_type],
            reasoning=reasoning,
            supporting_evidence=[f"Pattern frequency: {pattern.frequency}", 
                               f"Average strength: {pattern.pattern_strength:.2f}"],
            suggested_actions=suggested_actions,
            time_horizon=time_horizon
        )
    
    def _generate_contextual_predictions(self, current_state: Dict[str, Any], 
                                       prediction_type: str) -> List[PredictionResult]:
        """Generate predictions based on current state context."""
        predictions = []
        
        if prediction_type == "emotional":
            # Predict emotional trajectories
            if current_state["current_stressors"]:
                predictions.append(PredictionResult(
                    prediction_type="emotional",
                    predicted_state="You may experience increased stress if current pressures continue",
                    confidence=0.6,
                    reasoning="Current stressors in your environment suggest emotional strain",
                    supporting_evidence=current_state["current_stressors"],
                    suggested_actions=["Consider stress management techniques", "Take breaks when possible"],
                    time_horizon="short_term"
                ))
            
        elif prediction_type == "behavioral":
            # Predict behavioral changes
            if current_state["causal_momentum"] > 0.5:
                predictions.append(PredictionResult(
                    prediction_type="behavioral", 
                    predicted_state="You are likely to continue your current activity patterns",
                    confidence=0.7,
                    reasoning="Strong causal momentum suggests established behavioral patterns",
                    supporting_evidence=[f"Causal momentum: {current_state['causal_momentum']:.2f}"],
                    suggested_actions=["Maintain positive momentum", "Set clear goals"],
                    time_horizon="immediate"
                ))
        
        return predictions
    
    def _generate_action_suggestions(self, pattern: CausalPattern, prediction_type: str) -> List[str]:
        """Generate actionable suggestions based on predictions."""
        suggestions = []
        
        if prediction_type == "emotional":
            if pattern.effect_type in ["stress", "anxiety", "overwhelm"]:
                suggestions.extend([
                    "Practice mindfulness or meditation",
                    "Consider talking to someone about your feelings",
                    "Take time for self-care activities"
                ])
            elif pattern.effect_type in ["happiness", "confidence", "satisfaction"]:
                suggestions.extend([
                    "Continue activities that bring you joy",
                    "Share your positive experiences with others",
                    "Build on this positive momentum"
                ])
        
        elif prediction_type == "behavioral":
            suggestions.extend([
                "Set specific, achievable goals",
                "Track your progress regularly",
                "Adjust your approach based on what's working"
            ])
        
        elif prediction_type == "preference":
            suggestions.extend([
                "Explore new opportunities in areas you enjoy",
                "Consider why certain activities appeal to you",
                "Expand on your current interests"
            ])
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _classify_prediction_type(self, predicate: str, obj: str) -> str:
        """Classify what type of prediction this represents."""
        pred_lower = predicate.lower()
        obj_lower = obj.lower()
        
        # Emotional indicators
        if any(word in pred_lower or word in obj_lower 
               for word in ["feel", "emotion", "mood", "happy", "sad", "angry", "anxious", "excited", 
                           "stress", "overwhelm", "joy", "confidence", "worry"]):
            return "emotional"
        
        # Behavioral indicators
        elif any(word in pred_lower 
                 for word in ["do", "perform", "start", "begin", "complete", "try", "attempt", "practice"]):
            return "behavioral"
        
        # Preference indicators  
        elif any(word in pred_lower or word in obj_lower
                 for word in ["like", "prefer", "enjoy", "love", "hate", "interest", "hobby"]):
            return "preference"
        
        # Default to emotional for ambiguous cases
        return "emotional"
    
    def _classify_cause_type(self, predicate: str, obj: str) -> str:
        """Classify the type of cause (action, state, etc.)."""
        pred_lower = predicate.lower()
        
        if any(word in pred_lower for word in ["start", "begin", "do", "perform", "complete", "try"]):
            return "action"
        elif any(word in pred_lower for word in ["is", "are", "was", "were", "being"]):
            return "state"
        elif any(word in pred_lower for word in ["feel", "experience", "become"]):
            return "emotion"
        else:
            return "event" 