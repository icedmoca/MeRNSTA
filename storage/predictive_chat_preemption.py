#!/usr/bin/env python3
"""
Predictive Chat Preemption System for MeRNSTA

This system proactively responds in the chat UI with helpful suggestions when
prediction models forecast likely emotional or behavioral states.

📌 DO NOT HARDCODE trigger thresholds or message templates.
All parameters must be loaded from `config.settings` or environment config.
This is a zero-hardcoding cognitive subsystem.
"""

import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from storage.db_utils import get_connection_pool
from config.settings import DEFAULT_VALUES
from storage.causal_prediction_module import CausalPredictionModule


@dataclass
class ProactiveSuggestion:
    """Represents a proactive suggestion for the user."""
    suggestion_type: str  # "emotional_support", "productivity", "wellness", "goal_guidance"
    message: str
    confidence: float
    predicted_state: str
    suggested_actions: List[str]
    priority: int  # 1=high, 2=medium, 3=low
    trigger_reason: str


class PredictiveChatPreemption:
    """System for proactive chat interventions based on causal predictions."""
    
    def __init__(self):
        """Initialize the predictive chat preemption system."""
        # Load configuration parameters (no hardcoding)
        self.prediction_confidence_trigger = DEFAULT_VALUES.get("prediction_confidence_trigger", 0.7)
        self.emotional_intervention_threshold = DEFAULT_VALUES.get("emotional_intervention_threshold", 0.6)
        self.suggestion_cooldown_minutes = DEFAULT_VALUES.get("suggestion_cooldown_minutes", 30)
        self.max_suggestions_per_hour = DEFAULT_VALUES.get("max_proactive_suggestions_per_hour", 3)
        
        # Initialize prediction module
        self.prediction_module = CausalPredictionModule()
        
        # Message templates (should eventually be loaded from config)
        self.message_templates = self._load_message_templates()
        
        # Track suggestion history to avoid spam
        self.suggestion_history = {}
        
        print(f"[PredictiveChat] Initialized with confidence_trigger={self.prediction_confidence_trigger}, "
              f"intervention_threshold={self.emotional_intervention_threshold}")
    
    def _load_message_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Load message templates from configuration."""
        # TODO: Load from config file instead of hardcoding
        return {
            "emotional_support": {
                "exhausted": [
                    "I noticed you might be feeling drained. Would you like some suggestions for managing energy?",
                    "You've been working hard lately. Maybe it's time for a break? I can help you plan one.",
                    "Feeling exhausted is tough. Would you like to talk about what's been weighing on you?"
                ],
                "stressed": [
                    "It seems like stress might be building up. Want me to suggest some stress management techniques?",
                    "I can sense you might be feeling overwhelmed. Would breathing exercises or a quick walk help?",
                    "Stress can be challenging. Would you like some tips for breaking down what's bothering you?"
                ],
                "anxious": [
                    "I'm picking up signs you might be feeling anxious. Would grounding techniques be helpful right now?",
                    "Anxiety can be difficult to navigate. Want me to suggest some calming strategies?",
                    "Would it help to talk through what's making you feel anxious? I'm here to listen."
                ]
            },
            "productivity": {
                "procrastination": [
                    "Based on your patterns, you might be avoiding something challenging. Want help breaking it down?",
                    "Sometimes procrastination means the task feels overwhelming. Should we make it smaller?",
                    "I notice you might be stuck. Would the Pomodoro technique or a different approach help?"
                ],
                "focus_issues": [
                    "Your focus might be scattered today. Would you like suggestions for improving concentration?",
                    "Deep work can be challenging. Want me to help you create a focus plan?",
                    "Distractions happen to everyone. Should we set up some focus strategies together?"
                ]
            },
            "wellness": {
                "sleep_issues": [
                    "Your sleep patterns might be affecting your energy. Want some sleep hygiene tips?",
                    "Good rest is crucial for wellbeing. Would you like help creating a better bedtime routine?",
                    "Sleep troubles can impact everything. Should we explore some relaxation techniques?"
                ],
                "work_life_balance": [
                    "It seems like work might be taking over. Want to discuss boundary-setting strategies?",
                    "Balance is important for long-term success. Would you like help prioritizing?",
                    "Taking breaks isn't just nice—it's necessary. Should we plan some recovery time?"
                ]
            },
            "goal_guidance": {
                "motivation_low": [
                    "Your motivation might be dipping. Want to revisit your goals and find fresh inspiration?",
                    "Sometimes we need to reconnect with our 'why.' Would that conversation be helpful?",
                    "Motivation ebbs and flows. Should we explore what would re-energize you?"
                ],
                "progress_stalled": [
                    "Progress can sometimes plateau. Want to analyze what might move you forward?",
                    "Stalled progress is frustrating. Should we look at different approaches or strategies?",
                    "Sometimes a fresh perspective helps. Want to brainstorm new ways to approach your goals?"
                ]
            }
        }
    
    def check_for_proactive_suggestions(self, user_id: str, session_id: str = None) -> List[ProactiveSuggestion]:
        """
        Check if proactive suggestions should be triggered based on predictions.
        
        Args:
            user_id: User to check for
            session_id: Optional session context
            
        Returns:
            List of proactive suggestions to present to user
        """
        try:
            # Check rate limiting
            if not self._should_suggest_for_user(user_id):
                return []
            
            # Get predictions from causal prediction module
            predictions = self.prediction_module.predict_next_states(user_id, session_id)
            
            if not predictions:
                return []
            
            suggestions = []
            
            for prediction in predictions:
                # Check if prediction confidence exceeds trigger threshold
                if prediction.confidence >= self.prediction_confidence_trigger:
                    suggestion = self._generate_proactive_suggestion(prediction, user_id)
                    if suggestion:
                        suggestions.append(suggestion)
            
            # Sort by priority and confidence
            suggestions.sort(key=lambda s: (s.priority, -s.confidence))
            
            # Record suggestion attempt
            self._record_suggestion_attempt(user_id)
            
            return suggestions[:2]  # Limit to top 2 suggestions
            
        except Exception as e:
            print(f"[PredictiveChat] Error checking for suggestions: {e}")
            return []
    
    def _should_suggest_for_user(self, user_id: str) -> bool:
        """Check if we should suggest for this user based on rate limiting."""
        current_time = time.time()
        
        if user_id not in self.suggestion_history:
            return True
        
        user_history = self.suggestion_history[user_id]
        
        # Check cooldown period
        if current_time - user_history.get('last_suggestion', 0) < (self.suggestion_cooldown_minutes * 60):
            return False
        
        # Check hourly rate limit
        hour_start = current_time - 3600  # 1 hour ago
        recent_suggestions = [t for t in user_history.get('suggestion_times', []) if t > hour_start]
        
        if len(recent_suggestions) >= self.max_suggestions_per_hour:
            return False
        
        return True
    
    def _record_suggestion_attempt(self, user_id: str):
        """Record that we made a suggestion for rate limiting."""
        current_time = time.time()
        
        if user_id not in self.suggestion_history:
            self.suggestion_history[user_id] = {'suggestion_times': []}
        
        self.suggestion_history[user_id]['last_suggestion'] = current_time
        self.suggestion_history[user_id]['suggestion_times'].append(current_time)
        
        # Clean up old entries (older than 1 hour)
        hour_ago = current_time - 3600
        self.suggestion_history[user_id]['suggestion_times'] = [
            t for t in self.suggestion_history[user_id]['suggestion_times'] if t > hour_ago
        ]
    
    def _generate_proactive_suggestion(self, prediction, user_id: str) -> Optional[ProactiveSuggestion]:
        """Generate a proactive suggestion based on a prediction."""
        try:
            prediction_type = prediction.prediction_type
            predicted_state = prediction.predicted_state.lower()
            confidence = prediction.confidence
            
            # Determine suggestion category and specific state
            suggestion_type, specific_state = self._classify_suggestion_need(predicted_state, prediction_type)
            
            if not suggestion_type:
                return None
            
            # Check if this meets intervention threshold for emotional states
            if prediction_type == "emotional" and confidence < self.emotional_intervention_threshold:
                return None
            
            # Select appropriate message template
            template_category = self.message_templates.get(suggestion_type, {})
            message_options = template_category.get(specific_state, [])
            
            if not message_options:
                # Fallback to generic message
                message = f"Based on your recent patterns, you might be experiencing {predicted_state}. Would you like some suggestions to help?"
            else:
                # Select message based on confidence (higher confidence = more direct messages)
                message_index = min(len(message_options) - 1, int(confidence * len(message_options)))
                message = message_options[message_index]
            
            # Determine priority
            priority = self._calculate_suggestion_priority(prediction_type, specific_state, confidence)
            
            return ProactiveSuggestion(
                suggestion_type=suggestion_type,
                message=message,
                confidence=confidence,
                predicted_state=predicted_state,
                suggested_actions=prediction.suggested_actions or [],
                priority=priority,
                trigger_reason=f"{prediction_type} prediction: {predicted_state} (confidence: {confidence:.1%})"
            )
            
        except Exception as e:
            print(f"[PredictiveChat] Error generating suggestion: {e}")
            return None
    
    def _classify_suggestion_need(self, predicted_state: str, prediction_type: str) -> tuple:
        """Classify what type of suggestion is needed based on predicted state."""
        state_lower = predicted_state.lower()
        
        # Emotional support mapping
        if any(word in state_lower for word in ["exhausted", "tired", "drained", "fatigue"]):
            return ("emotional_support", "exhausted")
        elif any(word in state_lower for word in ["stressed", "overwhelm", "pressure"]):
            return ("emotional_support", "stressed") 
        elif any(word in state_lower for word in ["anxious", "worry", "nervous"]):
            return ("emotional_support", "anxious")
        
        # Productivity mapping
        elif any(word in state_lower for word in ["procrastinate", "avoid", "delay"]):
            return ("productivity", "procrastination")
        elif any(word in state_lower for word in ["unfocused", "distracted", "scattered"]):
            return ("productivity", "focus_issues")
        
        # Wellness mapping
        elif any(word in state_lower for word in ["sleep", "insomnia", "rest"]):
            return ("wellness", "sleep_issues")
        elif any(word in state_lower for word in ["balance", "overwork", "burnout"]):
            return ("wellness", "work_life_balance")
        
        # Goal guidance mapping
        elif any(word in state_lower for word in ["unmotivated", "uninspired", "stuck"]):
            return ("goal_guidance", "motivation_low")
        elif any(word in state_lower for word in ["stalled", "plateau", "blocked"]):
            return ("goal_guidance", "progress_stalled")
        
        # Default based on prediction type
        elif prediction_type == "emotional":
            return ("emotional_support", "general")
        elif prediction_type == "behavioral":
            return ("productivity", "general")
        
        return (None, None)
    
    def _calculate_suggestion_priority(self, prediction_type: str, specific_state: str, confidence: float) -> int:
        """Calculate priority for suggestion (1=high, 2=medium, 3=low)."""
        # High priority for emotional distress
        if prediction_type == "emotional" and specific_state in ["exhausted", "stressed", "anxious"]:
            return 1 if confidence > 0.8 else 2
        
        # Medium priority for productivity and wellness
        elif prediction_type in ["behavioral", "wellness"]:
            return 2 if confidence > 0.7 else 3
        
        # Lower priority for general suggestions
        else:
            return 3
    
    def format_suggestion_for_chat(self, suggestion: ProactiveSuggestion) -> Dict[str, Any]:
        """Format a suggestion for display in the chat UI."""
        return {
            "type": "proactive_suggestion",
            "message": suggestion.message,
            "confidence": round(suggestion.confidence, 2),
            "priority": suggestion.priority,
            "predicted_state": suggestion.predicted_state,
            "suggested_actions": suggestion.suggested_actions,
            "trigger_reason": suggestion.trigger_reason,
            "timestamp": time.time(),
            "ui_properties": {
                "show_dismiss": True,
                "show_more_info": True,
                "color_scheme": self._get_color_scheme_for_type(suggestion.suggestion_type),
                "icon": self._get_icon_for_type(suggestion.suggestion_type)
            }
        }
    
    def _get_color_scheme_for_type(self, suggestion_type: str) -> str:
        """Get color scheme for UI based on suggestion type."""
        color_schemes = {
            "emotional_support": "warm",  # Warm colors for emotional support
            "productivity": "blue",       # Blue for productivity
            "wellness": "green",          # Green for wellness
            "goal_guidance": "purple"     # Purple for goal guidance
        }
        return color_schemes.get(suggestion_type, "neutral")
    
    def _get_icon_for_type(self, suggestion_type: str) -> str:
        """Get icon for UI based on suggestion type."""
        icons = {
            "emotional_support": "💙",
            "productivity": "⚡",
            "wellness": "🌱",
            "goal_guidance": "🎯"
        }
        return icons.get(suggestion_type, "💡") 