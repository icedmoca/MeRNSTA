#!/usr/bin/env python3
"""
Emotion Model for MeRNSTA Phase 8 - Emotional State and Identity

This module implements emotional valence and arousal tracking across tokens and memory,
enabling mood-based decision making and emergent identity formation.
"""

import sqlite3
import logging
import json
import time
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class EmotionState:
    """Represents an emotional state with valence and arousal."""
    valence: float  # -1.0 (negative) to 1.0 (positive)
    arousal: float  # 0.0 (calm) to 1.0 (intense)
    timestamp: float = 0.0
    source_token_id: Optional[str] = None
    event_type: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        
        # Clamp values to valid ranges
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionState':
        return cls(**data)


@dataclass 
class MoodSignature:
    """Represents a mood signature derived from emotional patterns."""
    mood_label: str  # e.g., "calm", "tense", "curious", "frustrated"
    valence: float
    arousal: float
    confidence: float  # How confident we are in this mood assessment
    duration: float    # How long this mood has been active (seconds)
    dominant_emotions: List[str]  # Contributing emotion types
    timestamp: float = 0.0  # When this mood signature was created
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EmotionModel:
    """
    Models emotional valence and arousal across tokens and memory.
    
    Capabilities:
    - Track emotional state from token context and memory events
    - Calculate rolling mood state based on recent emotional history
    - Derive mood signatures and emotional patterns
    - Support emotional tagging of facts and goals
    - Enable mood-based decision modulation
    """
    
    def __init__(self, db_path: str = "enhanced_memory.db"):
        self.db_path = db_path
        
        # Emotion tracking configuration
        self.emotion_history_window = 3600  # 1 hour rolling window
        self.mood_calculation_window = 1800  # 30 min for mood calculation
        self.emotion_decay_rate = 0.95  # Exponential decay per hour
        self.arousal_baseline = 0.3  # Baseline arousal level
        self.valence_baseline = 0.1  # Slightly positive baseline
        
        # Emotion state tracking
        self.current_emotion_state = EmotionState(
            valence=self.valence_baseline,
            arousal=self.arousal_baseline
        )
        self.emotion_history: deque = deque(maxlen=1000)  # Recent emotions
        self.token_emotions: Dict[str, EmotionState] = {}  # Per-token emotions
        self.mood_history: List[MoodSignature] = []
        
        # Emotion event mappings
        self.emotion_event_mappings = {
            "contradiction": {"valence": -0.6, "arousal": 0.7, "label": "frustration"},
            "resolution": {"valence": 0.7, "arousal": 0.4, "label": "satisfaction"}, 
            "novelty": {"valence": 0.3, "arousal": 0.6, "label": "curiosity"},
            "confirmation": {"valence": 0.4, "arousal": 0.2, "label": "contentment"},
            "confusion": {"valence": -0.3, "arousal": 0.8, "label": "anxiety"},
            "discovery": {"valence": 0.8, "arousal": 0.7, "label": "excitement"},
            "conflict": {"valence": -0.5, "arousal": 0.9, "label": "tension"},
            "coherence": {"valence": 0.5, "arousal": 0.3, "label": "calm"},
            # Add missing emotion types from tests
            "excitement": {"valence": 0.8, "arousal": 0.7, "label": "excitement"},
            "curiosity": {"valence": 0.4, "arousal": 0.6, "label": "curiosity"},
            "frustration": {"valence": -0.6, "arousal": 0.7, "label": "frustration"},
            "satisfaction": {"valence": 0.7, "arousal": 0.4, "label": "satisfaction"}
        }
        
        # Mood classification mappings (valence, arousal -> mood)
        self.mood_classifications = [
            # High arousal moods
            ((-1.0, -0.3), (0.7, 1.0), "angry"),
            ((-0.3, 0.3), (0.7, 1.0), "tense"),
            ((0.3, 1.0), (0.7, 1.0), "excited"),
            
            # Medium arousal moods
            ((-1.0, -0.3), (0.3, 0.7), "frustrated"),
            ((-0.3, 0.3), (0.3, 0.7), "alert"),
            ((0.3, 1.0), (0.3, 0.7), "curious"),
            
            # Low arousal moods
            ((-1.0, -0.3), (0.0, 0.3), "sad"),
            ((-0.3, 0.3), (0.0, 0.3), "calm"),
            ((0.3, 1.0), (0.0, 0.3), "content")
        ]
        
        # Initialize database
        self._init_database()
        
        logger.info("[EmotionModel] Initialized emotion tracking system")
    
    def _init_database(self):
        """Initialize emotion tracking database tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Emotion states table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS emotion_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_id TEXT,
                    valence REAL NOT NULL,
                    arousal REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    event_type TEXT,
                    source_context TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            # Mood signatures table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mood_signatures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mood_label TEXT NOT NULL,
                    valence REAL NOT NULL,
                    arousal REAL NOT NULL,
                    confidence REAL NOT NULL,
                    duration REAL NOT NULL,
                    dominant_emotions TEXT,  -- JSON array
                    timestamp REAL NOT NULL,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            # Create indices for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_emotion_timestamp ON emotion_states(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_emotion_token ON emotion_states(token_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mood_timestamp ON mood_signatures(timestamp)")
            
            conn.commit()
            conn.close()
            
            logger.info("[EmotionModel] Initialized emotion database tables")
            
        except Exception as e:
            logger.error(f"Failed to initialize emotion database: {e}")
    
    def update_emotion_from_event(self, token_id: Optional[str], event_type: str, 
                                strength: float = 1.0, context: str = "") -> EmotionState:
        """
        Update emotional state based on a cognitive event.
        
        Args:
            token_id: Token associated with the event (can be None for global events)
            event_type: Type of event triggering emotion
            strength: Intensity multiplier (0.0 to 2.0)
            context: Additional context for the emotion
            
        Returns:
            The resulting emotion state
        """
        try:
            # Get base emotion mapping for event type
            if event_type not in self.emotion_event_mappings:
                logger.warning(f"Unknown emotion event type: {event_type}")
                event_type = "confusion"  # Default fallback
            
            base_emotion = self.emotion_event_mappings[event_type]
            
            # Apply strength multiplier with non-linear scaling
            adjusted_valence = base_emotion["valence"] * strength * 0.8  # Slight dampening
            adjusted_arousal = min(1.0, base_emotion["arousal"] * strength)
            
            # Create new emotion state
            emotion_state = EmotionState(
                valence=adjusted_valence,
                arousal=adjusted_arousal,
                source_token_id=token_id,
                event_type=event_type
            )
            
            # Store in history
            self.emotion_history.append(emotion_state)
            
            # Update token-specific emotion if token provided
            if token_id:
                self.token_emotions[token_id] = emotion_state
            
            # Update global current state with blending
            self._update_current_emotion_state(emotion_state)
            
            # Store in database
            self._store_emotion_state(emotion_state, context)
            
            logger.debug(f"[EmotionModel] Updated emotion from {event_type}: "
                        f"valence={emotion_state.valence:.2f}, arousal={emotion_state.arousal:.2f}")
            
            return emotion_state
            
        except Exception as e:
            logger.error(f"Failed to update emotion from event: {e}")
            return self.current_emotion_state
    
    def _update_current_emotion_state(self, new_emotion: EmotionState):
        """Update the global current emotion state by blending with new emotion."""
        # Time-based decay factor
        time_since_last = time.time() - self.current_emotion_state.timestamp
        decay_factor = math.exp(-time_since_last / 3600)  # 1-hour half-life
        
        # Adaptive blend weight - stronger emotions have more influence
        base_blend_weight = 0.5  # Increased from 0.3 for more responsiveness
        emotion_strength = abs(new_emotion.valence) + new_emotion.arousal
        blend_weight = min(0.8, base_blend_weight * (1 + emotion_strength))
        

        
        self.current_emotion_state.valence = (
            self.current_emotion_state.valence * (1 - blend_weight) * decay_factor +
            new_emotion.valence * blend_weight
        )
        
        self.current_emotion_state.arousal = (
            self.current_emotion_state.arousal * (1 - blend_weight) * decay_factor +
            new_emotion.arousal * blend_weight
        )
        
        self.current_emotion_state.timestamp = time.time()
        

    
    def get_current_mood(self) -> Dict[str, Any]:
        """
        Calculate current mood state from recent emotional history.
        
        Returns:
            Dictionary with mood information including label, valence, arousal, confidence
        """
        try:
            current_time = time.time()
            cutoff_time = current_time - self.mood_calculation_window
            
            # Filter recent emotions
            recent_emotions = [
                emotion for emotion in self.emotion_history
                if emotion.timestamp > cutoff_time
            ]
            
            if not recent_emotions:
                # Return baseline mood
                return {
                    "mood_label": "neutral",
                    "valence": self.valence_baseline,
                    "arousal": self.arousal_baseline,
                    "confidence": 0.3,
                    "duration": 0.0,
                    "contributing_events": []
                }
            
            # Calculate weighted averages (more recent = higher weight)
            total_weight = 0.0
            weighted_valence = 0.0
            weighted_arousal = 0.0
            event_counts = defaultdict(int)
            
            for emotion in recent_emotions:
                # Exponential decay weight based on recency
                age = current_time - emotion.timestamp
                weight = math.exp(-age / 1800)  # 30-minute half-life
                
                weighted_valence += emotion.valence * weight
                weighted_arousal += emotion.arousal * weight
                total_weight += weight
                
                if emotion.event_type:
                    event_counts[emotion.event_type] += 1
            
            # Normalize weighted averages
            avg_valence = weighted_valence / total_weight if total_weight > 0 else self.valence_baseline
            avg_arousal = weighted_arousal / total_weight if total_weight > 0 else self.arousal_baseline
            
            # Classify mood based on valence/arousal
            mood_label = self._classify_mood(avg_valence, avg_arousal)
            
            # Calculate confidence based on consistency and sample size
            confidence = min(1.0, len(recent_emotions) / 10.0)  # More samples = higher confidence
            
            # Calculate mood duration
            mood_duration = self._calculate_mood_duration(mood_label)
            
            mood_info = {
                "mood_label": mood_label,
                "valence": avg_valence,
                "arousal": avg_arousal, 
                "confidence": confidence,
                "duration": mood_duration,
                "contributing_events": dict(event_counts)
            }
            
            # Store mood signature
            self._store_mood_signature(mood_info)
            
            return mood_info
            
        except Exception as e:
            logger.error(f"Failed to calculate current mood: {e}")
            return {
                "mood_label": "unknown",
                "valence": 0.0,
                "arousal": 0.5,
                "confidence": 0.0,
                "duration": 0.0,
                "contributing_events": {}
            }
    
    def _classify_mood(self, valence: float, arousal: float) -> str:
        """Classify mood based on valence and arousal coordinates."""
        for (v_range, a_range, mood) in self.mood_classifications:
            if v_range[0] <= valence <= v_range[1] and a_range[0] <= arousal <= a_range[1]:
                return mood
        return "neutral"  # Default fallback
    
    def _calculate_mood_duration(self, current_mood: str) -> float:
        """Calculate how long the current mood has been active."""
        if not self.mood_history:
            return 0.0
        
        # Look backwards through mood history to find when current mood started
        current_time = time.time()
        for i in range(len(self.mood_history) - 1, -1, -1):
            if self.mood_history[i].mood_label != current_mood:
                # Found where mood changed
                if i + 1 < len(self.mood_history):
                    return current_time - self.mood_history[i + 1].timestamp
                break
        
        # If we didn't find a change, use the earliest mood record
        if self.mood_history:
            return current_time - self.mood_history[0].timestamp
        
        return 0.0
    
    def get_token_emotion(self, token_id: str) -> Tuple[float, float]:
        """
        Get emotional state for a specific token.
        
        Args:
            token_id: Token identifier
            
        Returns:
            Tuple of (valence, arousal)
        """
        if token_id in self.token_emotions:
            emotion = self.token_emotions[token_id]
            return (emotion.valence, emotion.arousal)
        
        # Return neutral emotion for unknown tokens
        return (0.0, 0.3)
    
    def get_mood_signature(self) -> str:
        """
        Generate a textual mood signature describing current emotional state.
        
        Returns:
            Human-readable mood signature string
        """
        try:
            mood_info = self.get_current_mood()
            
            mood_label = mood_info["mood_label"]
            confidence = mood_info["confidence"]
            duration = mood_info["duration"]
            events = mood_info["contributing_events"]
            
            # Format duration
            if duration < 60:
                duration_str = f"{duration:.0f}s"
            elif duration < 3600:
                duration_str = f"{duration/60:.0f}m"
            else:
                duration_str = f"{duration/3600:.1f}h"
            
            # Build signature string
            signature_parts = [f"{mood_label}"]
            
            if confidence > 0.7:
                signature_parts.append("(confident)")
            elif confidence < 0.3:
                signature_parts.append("(uncertain)")
            
            if duration > 300:  # More than 5 minutes
                signature_parts.append(f"for {duration_str}")
            
            # Add dominant events if significant
            if events:
                top_events = sorted(events.items(), key=lambda x: x[1], reverse=True)[:2]
                event_strs = [event for event, count in top_events if count >= 2]
                if event_strs:
                    signature_parts.append(f"from {', '.join(event_strs)}")
            
            return " ".join(signature_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate mood signature: {e}")
            return "neutral"
    
    def _store_emotion_state(self, emotion_state: EmotionState, context: str = ""):
        """Store emotion state in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO emotion_states 
                (token_id, valence, arousal, timestamp, event_type, source_context)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                emotion_state.source_token_id,
                emotion_state.valence,
                emotion_state.arousal,
                emotion_state.timestamp,
                emotion_state.event_type,
                context
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store emotion state: {e}")
    
    def _store_mood_signature(self, mood_info: Dict[str, Any]):
        """Store mood signature in database and history."""
        try:
            # Create mood signature object
            mood_signature = MoodSignature(
                mood_label=mood_info["mood_label"],
                valence=mood_info["valence"],
                arousal=mood_info["arousal"],
                confidence=mood_info["confidence"],
                duration=mood_info["duration"],
                dominant_emotions=list(mood_info["contributing_events"].keys())
            )
            
            # Add to history (limit size)
            self.mood_history.append(mood_signature)
            if len(self.mood_history) > 100:
                self.mood_history = self.mood_history[-100:]
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO mood_signatures 
                (mood_label, valence, arousal, confidence, duration, dominant_emotions, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                mood_signature.mood_label,
                mood_signature.valence,
                mood_signature.arousal,
                mood_signature.confidence,
                mood_signature.duration,
                json.dumps(mood_signature.dominant_emotions),
                time.time()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store mood signature: {e}")
    
    def get_emotional_history(self, token_id: Optional[str] = None, 
                            hours_back: float = 24.0) -> List[Dict[str, Any]]:
        """
        Get emotional history for analysis.
        
        Args:
            token_id: Optional token filter
            hours_back: How many hours of history to retrieve
            
        Returns:
            List of emotion history records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = time.time() - (hours_back * 3600)
            
            if token_id:
                cursor.execute("""
                    SELECT * FROM emotion_states 
                    WHERE token_id = ? AND timestamp > ?
                    ORDER BY timestamp DESC
                """, (token_id, cutoff_time))
            else:
                cursor.execute("""
                    SELECT * FROM emotion_states
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                """, (cutoff_time,))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to dictionaries
            columns = ["id", "token_id", "valence", "arousal", "timestamp", 
                      "event_type", "source_context", "created_at"]
            
            return [dict(zip(columns, row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get emotional history: {e}")
            return []
    
    def set_emotion_override(self, valence: float, arousal: float, duration: float = 300.0):
        """
        Manually override emotion state for debugging/testing.
        
        Args:
            valence: Valence value (-1.0 to 1.0)
            arousal: Arousal value (0.0 to 1.0)  
            duration: How long override lasts in seconds
        """
        try:
            override_emotion = EmotionState(
                valence=valence,
                arousal=arousal,
                event_type="manual_override"
            )
            
            # Apply multiple times to override recent history
            for _ in range(5):
                self.emotion_history.append(override_emotion)
            
            self.current_emotion_state = override_emotion
            
            logger.info(f"[EmotionModel] Applied emotion override: valence={valence}, arousal={arousal}")
            
        except Exception as e:
            logger.error(f"Failed to set emotion override: {e}")


def get_emotion_model(db_path: str = "enhanced_memory.db") -> EmotionModel:
    """Get or create the global emotion model instance."""
    # Use db_path in instance key to support different databases (e.g., for testing)
    instance_key = f'_instance_{db_path}'
    if not hasattr(get_emotion_model, instance_key):
        setattr(get_emotion_model, instance_key, EmotionModel(db_path))
    return getattr(get_emotion_model, instance_key)