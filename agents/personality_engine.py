#!/usr/bin/env python3
"""
Expressive Personality Engine for MeRNSTA Phase 9

This module transforms MeRNSTA from a reactive responder to an expressive agent
with dynamic personality that fuses memory, mood, identity, and context into
emotionally resonant language.
"""

import logging
import re
import random
import time
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ResponseMode(Enum):
    """Response modes based on emotional state and context."""
    RATIONAL = "rational"
    EMPATHETIC = "empathetic" 
    PLAYFUL = "playful"
    ASSERTIVE = "assertive"


@dataclass
class PersonalityEvolutionEvent:
    """Represents a single personality evolution event."""
    timestamp: str
    trigger: str  # 'feedback', 'manual', 'automatic', 'emotional_dissonance'
    changes: Dict[str, Any]  # What changed
    reason: str  # Why it changed
    updated_by: str  # What triggered the change
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PersonalityState:
    """Tracks current personality state with evolution capabilities."""
    # Current state
    tone: str  # Current dominant tone/mode
    emotional_state: str  # Current emotional baseline
    core_traits: Dict[str, float]  # Trait name -> strength (0.0-1.0)
    
    # Evolution tracking
    evolution_history: List[PersonalityEvolutionEvent]
    last_updated: str
    last_updated_by: str
    feedback_sensitivity: float  # How responsive to feedback (0.0-1.0)
    stability_factor: float  # Resistance to change (0.0-1.0)
    
    # Metadata
    creation_timestamp: str
    total_evolutions: int
    
    def evolve_from_feedback(self, feedback: str, trigger_source: str = "user_feedback") -> bool:
        """
        Evolve personality traits based on feedback.
        
        Args:
            feedback: Text feedback describing desired changes
            trigger_source: What triggered this evolution
            
        Returns:
            True if personality evolved, False if no change needed
        """
        try:
            changes_made = {}
            feedback_lower = feedback.lower()
            
            # Analyze feedback for trait adjustments
            trait_adjustments = self._analyze_feedback_for_traits(feedback_lower)
            
            # Apply trait changes with sensitivity and stability factors
            for trait_name, adjustment in trait_adjustments.items():
                if trait_name in self.core_traits:
                    old_value = self.core_traits[trait_name]
                    # Apply adjustment scaled by sensitivity and damped by stability
                    change_amount = adjustment * self.feedback_sensitivity * (1.0 - self.stability_factor)
                    new_value = max(0.0, min(1.0, old_value + change_amount))
                    
                    if abs(new_value - old_value) > 0.05:  # Minimum change threshold
                        self.core_traits[trait_name] = new_value
                        changes_made[trait_name] = {"from": old_value, "to": new_value}
                else:
                    # Add new trait if suggested strongly
                    if abs(adjustment) > 0.3:
                        initial_strength = min(0.5, abs(adjustment) * self.feedback_sensitivity)
                        self.core_traits[trait_name] = initial_strength
                        changes_made[trait_name] = {"from": 0.0, "to": initial_strength}
            
            # Update emotional state if feedback suggests it
            new_emotional_state = self._detect_emotional_state_from_feedback(feedback_lower)
            if new_emotional_state and new_emotional_state != self.emotional_state:
                changes_made["emotional_state"] = {"from": self.emotional_state, "to": new_emotional_state}
                self.emotional_state = new_emotional_state
            
            # Update tone if feedback suggests it
            new_tone = self._detect_tone_from_feedback(feedback_lower)
            if new_tone and new_tone != self.tone:
                changes_made["tone"] = {"from": self.tone, "to": new_tone}
                self.tone = new_tone
            
            # Record evolution if any changes were made
            if changes_made:
                evolution_event = PersonalityEvolutionEvent(
                    timestamp=datetime.now().isoformat(),
                    trigger="feedback",
                    changes=changes_made,
                    reason=feedback[:200],  # Truncate long feedback
                    updated_by=trigger_source
                )
                
                self.evolution_history.append(evolution_event)
                self.last_updated = evolution_event.timestamp
                self.last_updated_by = trigger_source
                self.total_evolutions += 1
                
                logger.info(f"[PersonalityState] Evolved from feedback: {len(changes_made)} changes made")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"[PersonalityState] Error evolving from feedback: {e}")
            return False
    
    def _analyze_feedback_for_traits(self, feedback: str) -> Dict[str, float]:
        """Analyze feedback text to suggest trait adjustments."""
        adjustments = {}
        
        # Positive trait suggestions
        if any(word in feedback for word in ["empathetic", "understanding", "caring", "compassionate"]):
            adjustments["empathetic"] = 0.2
        if any(word in feedback for word in ["analytical", "logical", "rational", "precise"]):
            adjustments["analytical"] = 0.2
        if any(word in feedback for word in ["curious", "exploring", "questioning", "wondering"]):
            adjustments["curious"] = 0.2
        if any(word in feedback for word in ["optimistic", "positive", "hopeful", "encouraging"]):
            adjustments["optimistic"] = 0.2
        if any(word in feedback for word in ["playful", "fun", "engaging", "energetic"]):
            adjustments["playful"] = 0.2
        if any(word in feedback for word in ["confident", "assertive", "direct", "strong"]):
            adjustments["confident"] = 0.2
        
        # Negative trait adjustments
        if any(word in feedback for word in ["too cold", "harsh", "robotic", "unfriendly"]):
            adjustments["empathetic"] = 0.3
            adjustments["analytical"] = -0.2
        if any(word in feedback for word in ["too emotional", "too caring", "overwhelming"]):
            adjustments["empathetic"] = -0.2
            adjustments["analytical"] = 0.2
        if any(word in feedback for word in ["boring", "dull", "lifeless"]):
            adjustments["playful"] = 0.3
            adjustments["curious"] = 0.2
        if any(word in feedback for word in ["too excited", "too much", "overwhelming"]):
            adjustments["playful"] = -0.2
        
        return adjustments
    
    def _detect_emotional_state_from_feedback(self, feedback: str) -> Optional[str]:
        """Detect desired emotional state from feedback."""
        if any(word in feedback for word in ["calmer", "peaceful", "serene"]):
            return "calm"
        elif any(word in feedback for word in ["excited", "energetic", "enthusiastic"]):
            return "excited"
        elif any(word in feedback for word in ["focused", "serious", "concentrated"]):
            return "focused"
        elif any(word in feedback for word in ["warm", "friendly", "welcoming"]):
            return "warm"
        return None
    
    def _detect_tone_from_feedback(self, feedback: str) -> Optional[str]:
        """Detect desired tone from feedback."""
        if any(word in feedback for word in ["rational", "logical", "analytical"]):
            return "rational"
        elif any(word in feedback for word in ["empathetic", "caring", "understanding"]):
            return "empathetic"
        elif any(word in feedback for word in ["playful", "fun", "casual"]):
            return "playful"
        elif any(word in feedback for word in ["assertive", "direct", "confident"]):
            return "assertive"
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tone": self.tone,
            "emotional_state": self.emotional_state,
            "core_traits": self.core_traits,
            "evolution_history": [event.to_dict() for event in self.evolution_history],
            "last_updated": self.last_updated,
            "last_updated_by": self.last_updated_by,
            "feedback_sensitivity": self.feedback_sensitivity,
            "stability_factor": self.stability_factor,
            "creation_timestamp": self.creation_timestamp,
            "total_evolutions": self.total_evolutions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonalityState":
        """Create from dictionary (JSON deserialization)."""
        evolution_history = [
            PersonalityEvolutionEvent(**event) for event in data.get("evolution_history", [])
        ]
        
        return cls(
            tone=data.get("tone", "rational"),
            emotional_state=data.get("emotional_state", "neutral"),
            core_traits=data.get("core_traits", {}),
            evolution_history=evolution_history,
            last_updated=data.get("last_updated", datetime.now().isoformat()),
            last_updated_by=data.get("last_updated_by", "system"),
            feedback_sensitivity=data.get("feedback_sensitivity", 0.7),
            stability_factor=data.get("stability_factor", 0.3),
            creation_timestamp=data.get("creation_timestamp", datetime.now().isoformat()),
            total_evolutions=data.get("total_evolutions", 0)
        )


@dataclass
class ToneProfile:
    """Represents the current personality tone configuration."""
    mode: ResponseMode
    primary_traits: List[str]  # Dominant identity traits
    mood_label: str  # Current mood
    emotional_intensity: float  # 0.0 to 1.0
    confidence_level: float  # Based on identity confidence
    conversational_energy: float  # Based on arousal
    valence_bias: float  # Positive/negative tendency
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "primary_traits": self.primary_traits,
            "mood_label": self.mood_label,
            "emotional_intensity": self.emotional_intensity,
            "confidence_level": self.confidence_level,
            "conversational_energy": self.conversational_energy,
            "valence_bias": self.valence_bias
        }
    
    def get_description(self) -> str:
        """Generate human-readable description of current tone."""
        trait_str = ", ".join(self.primary_traits[:3]) if self.primary_traits else "developing"
        
        energy_desc = ""
        if self.conversational_energy > 0.7:
            energy_desc = "energetic"
        elif self.conversational_energy > 0.4:
            energy_desc = "engaged"
        else:
            energy_desc = "calm"
        
        return f"{self.mode.value.title()} • {trait_str} • {self.mood_label} • {energy_desc}"


class ExpressivePersonalityEngine:
    """
    Core engine that transforms basic responses into personality-infused expressions.
    
    Capabilities:
    - Dynamic response mode selection based on emotional state
    - Identity trait integration into conversational style
    - Mood-based tone modulation
    - Context-aware personality adaptation
    - Response styling with emotional resonance
    - Dynamic personality evolution from feedback
    - Persistent personality state with history tracking
    - Automatic evolution triggers for emotional dissonance
    """
    
    def __init__(self, db_path: str = "enhanced_memory.db"):
        self.db_path = db_path
        
        # Response mode override (None = automatic selection)
        self.mode_override: Optional[ResponseMode] = None
        
        # Personality configuration
        self.trait_influence_weight = 0.6  # How much identity affects tone
        self.mood_influence_weight = 0.4   # How much current mood affects tone
        self.energy_scaling_factor = 0.8   # Scaling for conversational energy
        
        # Persistent personality state
        self.personality_file = Path("output") / "personality.json"
        self.personality_state = self._load_personality_state()
        
        # Response transformation templates
        self._init_response_templates()
        
        # Personality quirks and linguistic patterns
        self._init_personality_patterns()
        
        logger.info("[ExpressivePersonalityEngine] Initialized personality engine with persistent state")
    
    def _init_response_templates(self):
        """Initialize response transformation templates for each mode."""
        
        self.response_templates = {
            ResponseMode.RATIONAL: {
                "prefixes": [
                    "Based on my analysis, ",
                    "From what I understand, ",
                    "The data suggests that ",
                    "Logically speaking, ",
                    "My assessment is that "
                ],
                "connectors": [
                    "therefore", "consequently", "as a result", 
                    "which means", "indicating that"
                ],
                "hedges": [
                    "it appears that", "the evidence suggests", 
                    "most likely", "in all probability"
                ],
                "certainty_boosters": [
                    "clearly", "obviously", "undoubtedly", 
                    "without question", "definitively"
                ],
                "style_markers": {
                    "technical_terms": True,
                    "formal_language": True,
                    "structured_responses": True,
                    "evidence_focus": True
                }
            },
            
            ResponseMode.EMPATHETIC: {
                "prefixes": [
                    "I understand that ",
                    "I can see how ",
                    "That sounds like ",
                    "I imagine that must be ",
                    "I appreciate that "
                ],
                "emotional_acknowledgments": [
                    "that must be challenging", "I can relate to that feeling",
                    "that's completely understandable", "your perspective makes sense",
                    "I hear what you're saying"
                ],
                "supportive_phrases": [
                    "you're not alone in feeling this way",
                    "it's natural to feel that way",
                    "your feelings are valid",
                    "I'm here to help work through this"
                ],
                "gentle_transitions": [
                    "perhaps we could explore", "what if we considered",
                    "another way to look at this might be", "it might help to think about"
                ],
                "style_markers": {
                    "emotional_language": True,
                    "gentle_pacing": True,
                    "validation_focus": True,
                    "personal_pronouns": True
                }
            },
            
            ResponseMode.PLAYFUL: {
                "prefixes": [
                    "Oh, that's interesting! ",
                    "Hmm, this is curious... ",
                    "Well, well, well... ",
                    "Here's a fun thought: ",
                    "You know what's fascinating? "
                ],
                "curiosity_hooks": [
                    "I wonder if...", "What if we...", "Here's a wild idea:",
                    "Let's explore this...", "This makes me think..."
                ],
                "playful_connectors": [
                    "and here's the kicker", "plot twist", "but wait, there's more",
                    "here's where it gets interesting", "now this is where things get fun"
                ],
                "enthusiasm_markers": [
                    "!", "...", "really", "actually", "totally",
                    "absolutely", "definitely", "for sure"
                ],
                "style_markers": {
                    "casual_language": True,
                    "enthusiasm": True,
                    "curiosity_expression": True,
                    "informal_tone": True
                }
            },
            
            ResponseMode.ASSERTIVE: {
                "prefixes": [
                    "Let me be clear: ",
                    "The reality is ",
                    "Here's what's happening: ",
                    "I need to point out that ",
                    "This is important: "
                ],
                "direct_statements": [
                    "simply put", "bottom line", "the truth is",
                    "without beating around the bush", "to be direct"
                ],
                "emphasis_markers": [
                    "absolutely", "completely", "entirely", 
                    "without question", "undeniably"
                ],
                "corrective_phrases": [
                    "actually", "in fact", "to be precise",
                    "let me correct that", "more accurately"
                ],
                "style_markers": {
                    "direct_language": True,
                    "confident_tone": True,
                    "concise_responses": True,
                    "strong_statements": True
                }
            }
        }
    
    def _init_personality_patterns(self):
        """Initialize personality-specific linguistic patterns."""
        
        # Trait-based language patterns
        self.trait_patterns = {
            "curious": {
                "questions": ["What if...?", "I wonder...", "How might...?"],
                "explorations": ["Let's explore", "This leads me to wonder", "I'm curious about"],
                "connectors": ["which makes me think", "this brings up", "I'm wondering if"]
            },
            "analytical": {
                "frameworks": ["Breaking this down", "Analyzing this", "Looking at the components"],
                "logic_chains": ["This suggests", "Following this logic", "The implications are"],
                "precision": ["To be precise", "More specifically", "Drilling down"]
            },
            "empathetic": {
                "understanding": ["I can understand", "I see how", "That makes sense"],
                "validation": ["Your feelings are valid", "That's understandable", "I hear you"],
                "support": ["I'm here to help", "We can work through this", "You're not alone"]
            },
            "skeptical": {
                "questioning": ["I question whether", "Is it possible that", "We should consider"],
                "alternatives": ["On the other hand", "Another perspective", "What if instead"],
                "caution": ["I'm hesitant to assume", "We should be careful", "Let's not jump to conclusions"]
            },
            "optimistic": {
                "positive_framing": ["The good news is", "Looking on the bright side", "What's encouraging"],
                "possibilities": ["There's potential for", "We could see", "This opens up"],
                "hope": ["I'm hopeful that", "There's reason to believe", "Things could improve"]
            },
            "resilient": {
                "perseverance": ["Despite the challenges", "We can overcome this", "Let's push through"],
                "strength": ["Building on what works", "Finding our footing", "Staying strong"],
                "adaptation": ["Let's adapt", "We can adjust", "Finding another way"]
            }
        }
    
    def get_current_tone_profile(self) -> ToneProfile:
        """
        Generate current tone profile based on emotional state and identity.
        
        Returns:
            ToneProfile object describing current personality configuration
        """
        try:
            # Import models with lazy loading
            from storage.emotion_model import get_emotion_model
            from storage.self_model import get_self_aware_model
            
            emotion_model = get_emotion_model(self.db_path)
            self_model = get_self_aware_model(self.db_path)
            
            # Get current emotional state
            current_mood = emotion_model.get_current_mood()
            mood_label = current_mood.get("mood_label", "neutral")
            valence = current_mood.get("valence", 0.0)
            arousal = current_mood.get("arousal", 0.3)
            confidence = current_mood.get("confidence", 0.5)
            
            # Get identity traits
            identity_traits = self_model.identity_traits
            primary_traits = []
            avg_trait_confidence = 0.0
            
            if identity_traits:
                # Sort traits by strength * confidence
                sorted_traits = sorted(
                    identity_traits.items(),
                    key=lambda x: x[1].strength * x[1].confidence,
                    reverse=True
                )
                
                # Take top 3 significant traits
                for trait_name, trait in sorted_traits[:3]:
                    if trait.confidence >= 0.4 and trait.strength >= 0.3:
                        primary_traits.append(trait_name)
                
                # Calculate average trait confidence
                confident_traits = [t for _, t in sorted_traits if t.confidence >= 0.4]
                if confident_traits:
                    avg_trait_confidence = sum(t.confidence for t in confident_traits) / len(confident_traits)
            
            # Determine response mode
            response_mode = self._determine_response_mode(valence, arousal, primary_traits, mood_label)
            
            # Create tone profile
            tone_profile = ToneProfile(
                mode=response_mode,
                primary_traits=primary_traits,
                mood_label=mood_label,
                emotional_intensity=abs(valence) + arousal * 0.5,  # Combined intensity
                confidence_level=avg_trait_confidence,
                conversational_energy=arousal,
                valence_bias=valence
            )
            
            return tone_profile
            
        except Exception as e:
            print(f"[ExpressivePersonalityEngine] Error generating tone profile: {e}")
            
            # Return default neutral profile
            return ToneProfile(
                mode=ResponseMode.RATIONAL,
                primary_traits=["developing"],
                mood_label="neutral",
                emotional_intensity=0.3,
                confidence_level=0.5,
                conversational_energy=0.3,
                valence_bias=0.0
            )
    
    def _determine_response_mode(self, valence: float, arousal: float, 
                               traits: List[str], mood_label: str) -> ResponseMode:
        """
        Determine appropriate response mode based on emotional state and traits.
        
        Args:
            valence: Emotional valence (-1.0 to 1.0)
            arousal: Emotional arousal (0.0 to 1.0)
            traits: Primary identity traits
            mood_label: Current mood label
            
        Returns:
            Selected ResponseMode
        """
        # Use override if set
        if self.mode_override:
            return self.mode_override
        
        # Mood-based mode selection with trait modifiers
        if mood_label in ["angry", "frustrated", "tense"] or (valence < -0.3 and arousal > 0.6):
            # High negative arousal -> assertive
            return ResponseMode.ASSERTIVE
        
        elif mood_label in ["excited", "curious"] or (valence > 0.5 and arousal > 0.6):
            # High positive arousal -> playful (unless analytical trait is very strong)
            if "analytical" in traits and "curious" not in traits[:2]:
                return ResponseMode.RATIONAL
            else:
                return ResponseMode.PLAYFUL
        
        elif mood_label in ["sad", "content", "calm"] or (arousal < 0.4):
            # Low arousal states
            if "empathetic" in traits or valence < -0.2:
                return ResponseMode.EMPATHETIC
            else:
                return ResponseMode.RATIONAL
        
        else:
            # Default based on strongest traits
            if traits:
                dominant_trait = traits[0]
                if dominant_trait in ["empathetic"]:
                    return ResponseMode.EMPATHETIC
                elif dominant_trait in ["curious"] and valence > 0:
                    return ResponseMode.PLAYFUL
                elif dominant_trait in ["analytical", "skeptical", "resilient"]:
                    return ResponseMode.RATIONAL
            
            # Final fallback
            return ResponseMode.RATIONAL
    
    def style_response(self, text: str, context: Dict[str, Any] = None) -> str:
        """
        Transform base response text with personality and emotional tone.
        
        Args:
            text: Base response text to transform
            context: Additional context for tone adjustment
            
        Returns:
            Personality-styled response text
        """
        try:
            if not text or not text.strip():
                return text
            
            # Get current tone profile
            tone_profile = self.get_current_tone_profile()
            
            # Apply context-specific adjustments
            if context:
                tone_profile = self._adjust_tone_for_context(tone_profile, context)
            
            # Apply personality styling
            styled_text = self._apply_personality_styling(text, tone_profile)
            
            return styled_text
            
        except Exception as e:
            print(f"[ExpressivePersonalityEngine] Error styling response: {e}")
            return text  # Return original text if styling fails
    
    def _adjust_tone_for_context(self, tone_profile: ToneProfile, context: Dict[str, Any]) -> ToneProfile:
        """Adjust tone profile based on conversational context."""
        
        # Create a copy to avoid modifying original
        adjusted_profile = ToneProfile(
            mode=tone_profile.mode,
            primary_traits=tone_profile.primary_traits.copy(),
            mood_label=tone_profile.mood_label,
            emotional_intensity=tone_profile.emotional_intensity,
            confidence_level=tone_profile.confidence_level,
            conversational_energy=tone_profile.conversational_energy,
            valence_bias=tone_profile.valence_bias
        )
        
        # Context-specific adjustments
        if context.get("contradiction_detected"):
            # Contradiction detected -> more analytical/assertive
            if adjusted_profile.mode == ResponseMode.PLAYFUL:
                adjusted_profile.mode = ResponseMode.RATIONAL
            adjusted_profile.emotional_intensity += 0.2
        
        if context.get("user_emotion") == "frustrated":
            # User frustrated -> more empathetic
            if adjusted_profile.mode != ResponseMode.ASSERTIVE:
                adjusted_profile.mode = ResponseMode.EMPATHETIC
        
        if context.get("complex_query"):
            # Complex query -> more rational
            if adjusted_profile.mode == ResponseMode.PLAYFUL:
                adjusted_profile.mode = ResponseMode.RATIONAL
        
        if context.get("casual_conversation"):
            # Casual conversation -> more playful if possible
            if adjusted_profile.mode == ResponseMode.RATIONAL and adjusted_profile.valence_bias > 0:
                adjusted_profile.mode = ResponseMode.PLAYFUL
        
        return adjusted_profile
    
    def _apply_personality_styling(self, text: str, tone_profile: ToneProfile) -> str:
        """Apply personality-specific styling to response text."""
        
        styled_text = text
        mode = tone_profile.mode
        templates = self.response_templates[mode]
        
        # Apply mode-specific transformations
        if mode == ResponseMode.RATIONAL:
            styled_text = self._apply_rational_styling(styled_text, tone_profile, templates)
        elif mode == ResponseMode.EMPATHETIC:
            styled_text = self._apply_empathetic_styling(styled_text, tone_profile, templates)
        elif mode == ResponseMode.PLAYFUL:
            styled_text = self._apply_playful_styling(styled_text, tone_profile, templates)
        elif mode == ResponseMode.ASSERTIVE:
            styled_text = self._apply_assertive_styling(styled_text, tone_profile, templates)
        
        # Apply trait-specific language patterns
        styled_text = self._apply_trait_patterns(styled_text, tone_profile)
        
        # Apply emotional intensity adjustments
        styled_text = self._apply_emotional_intensity(styled_text, tone_profile)
        
        return styled_text
    
    def _apply_rational_styling(self, text: str, tone_profile: ToneProfile, templates: Dict) -> str:
        """Apply rational mode styling."""
        
        # Add analytical prefixes for longer responses
        if len(text) > 50 and not any(text.startswith(p) for p in templates["prefixes"]):
            if tone_profile.confidence_level > 0.7:
                prefix = random.choice(templates["prefixes"])
                text = prefix + text.lower()
        
        # Add hedging for uncertain statements
        if tone_profile.confidence_level < 0.6:
            hedge_phrases = templates["hedges"]
            # Look for definitive statements and soften them
            for hedge in hedge_phrases:
                if random.random() < 0.3:  # 30% chance to add hedging
                    text = re.sub(r'\b(is|are|will|must)\b', f'{hedge} \\1', text, count=1)
                    break
        
        # Add logical connectors
        sentences = text.split('. ')
        if len(sentences) > 1:
            connector = random.choice(templates["connectors"])
            sentences[1] = connector + ", " + sentences[1].lower()
            text = '. '.join(sentences)
        
        return text
    
    def _apply_empathetic_styling(self, text: str, tone_profile: ToneProfile, templates: Dict) -> str:
        """Apply empathetic mode styling."""
        
        # Add emotional acknowledgment
        if tone_profile.valence_bias < 0:  # Negative emotional context
            acknowledgment = random.choice(templates["emotional_acknowledgments"])
            if not text.startswith(("I understand", "I can see", "That sounds")):
                text = f"I understand that {acknowledgment.lower()}. {text}"
        
        # Add supportive phrases for challenging topics
        if any(word in text.lower() for word in ["difficult", "hard", "problem", "issue", "struggle"]):
            support = random.choice(templates["supportive_phrases"])
            text += f" {support.capitalize()}."
        
        # Soften direct statements
        text = re.sub(r'\bYou should\b', 'You might consider', text)
        text = re.sub(r'\bYou need to\b', 'It might help to', text)
        text = re.sub(r'\bYou must\b', 'It would be good to', text)
        
        return text
    
    def _apply_playful_styling(self, text: str, tone_profile: ToneProfile, templates: Dict) -> str:
        """Apply playful mode styling."""
        
        # Add enthusiasm markers
        if tone_profile.conversational_energy > 0.6:
            # Occasionally add exclamation points
            if random.random() < 0.4 and not text.endswith('!'):
                text = text.rstrip('.') + '!'
        
        # Add curiosity hooks
        if "interesting" in text.lower() or "curious" in text.lower():
            hook = random.choice(templates["curiosity_hooks"])
            text += f" {hook}"
        
        # Make language more casual
        text = re.sub(r'\bHowever,\s*', 'But ', text)
        text = re.sub(r'\bTherefore,\s*', 'So ', text)
        text = re.sub(r'\bNevertheless,\s*', 'Still ', text)
        
        # Add playful connectors between sentences
        sentences = text.split('. ')
        if len(sentences) > 1 and random.random() < 0.3:
            connector = random.choice(templates["playful_connectors"])
            sentences[1] = f"{connector} - " + sentences[1].lower()
            text = '. '.join(sentences)
        
        return text
    
    def _apply_assertive_styling(self, text: str, tone_profile: ToneProfile, templates: Dict) -> str:
        """Apply assertive mode styling."""
        
        # Add direct prefixes for important statements
        if tone_profile.emotional_intensity > 0.7:
            prefix = random.choice(templates["prefixes"])
            if not any(text.startswith(p.split(':')[0]) for p in templates["prefixes"]):
                text = prefix + text.lower()
        
        # Strengthen statements with emphasis
        emphasis = random.choice(templates["emphasis_markers"])
        # Find weak qualifiers and strengthen them
        text = re.sub(r'\bmight be\b', f'{emphasis} is', text)
        text = re.sub(r'\bcould be\b', f'{emphasis} is', text)
        text = re.sub(r'\bseems to be\b', f'{emphasis} is', text)
        
        # Make corrections more direct
        if "actually" in text.lower() or "correction" in text.lower():
            corrective = random.choice(templates["corrective_phrases"])
            text = re.sub(r'\bActually,\b', f'{corrective.capitalize()},', text)
        
        # Remove hedging language
        text = re.sub(r'\bI think that\b', '', text)
        text = re.sub(r'\bperhaps\b', '', text)
        text = re.sub(r'\bmaybe\b', '', text)
        
        return text
    
    def _apply_trait_patterns(self, text: str, tone_profile: ToneProfile) -> str:
        """Apply identity trait-specific language patterns."""
        
        for trait in tone_profile.primary_traits[:2]:  # Apply top 2 traits
            if trait in self.trait_patterns:
                patterns = self.trait_patterns[trait]
                
                # Apply trait-specific connectors occasionally
                if random.random() < 0.3:
                    for pattern_type, phrases in patterns.items():
                        if pattern_type == "connectors" and ". " in text:
                            sentences = text.split('. ')
                            if len(sentences) > 1:
                                connector = random.choice(phrases)
                                sentences[1] = f"{connector}, " + sentences[1].lower()
                                text = '. '.join(sentences)
                                break
        
        return text
    
    def _apply_emotional_intensity(self, text: str, tone_profile: ToneProfile) -> str:
        """Apply emotional intensity adjustments to text."""
        
        intensity = tone_profile.emotional_intensity
        
        # High intensity: more emphatic language
        if intensity > 0.7:
            # Add emphasis through repetition or stronger words
            text = re.sub(r'\bvery\b', 'extremely', text)
            text = re.sub(r'\bgood\b', 'excellent', text)
            text = re.sub(r'\bbad\b', 'terrible', text)
        
        # Low intensity: softer language
        elif intensity < 0.3:
            # Soften strong statements
            text = re.sub(r'\bmust\b', 'should probably', text)
            text = re.sub(r'\bnever\b', 'rarely', text)
            text = re.sub(r'\balways\b', 'usually', text)
        
        return text
    
    def _load_personality_state(self) -> PersonalityState:
        """Load personality state from persistent storage."""
        try:
            if self.personality_file.exists():
                with open(self.personality_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"[ExpressivePersonalityEngine] Loaded personality state from {self.personality_file}")
                    return PersonalityState.from_dict(data)
        except Exception as e:
            logger.warning(f"[ExpressivePersonalityEngine] Error loading personality state: {e}")
        
        # Create default personality state
        logger.info("[ExpressivePersonalityEngine] Creating default personality state")
        return PersonalityState(
            tone="rational",
            emotional_state="neutral",
            core_traits={
                "curious": 0.7,
                "analytical": 0.8,
                "helpful": 0.9,
                "empathetic": 0.6
            },
            evolution_history=[],
            last_updated=datetime.now().isoformat(),
            last_updated_by="system_init",
            feedback_sensitivity=0.7,
            stability_factor=0.3,
            creation_timestamp=datetime.now().isoformat(),
            total_evolutions=0
        )
    
    def _save_personality_state(self) -> bool:
        """Save personality state to persistent storage."""
        try:
            # Ensure output directory exists
            self.personality_file.parent.mkdir(exist_ok=True)
            
            with open(self.personality_file, 'w', encoding='utf-8') as f:
                json.dump(self.personality_state.to_dict(), f, indent=2)
            
            logger.info(f"[ExpressivePersonalityEngine] Saved personality state to {self.personality_file}")
            return True
            
        except Exception as e:
            logger.error(f"[ExpressivePersonalityEngine] Error saving personality state: {e}")
            return False
    
    def evolve_personality(self, feedback: str, trigger_source: str = "user_feedback") -> bool:
        """
        Evolve personality based on feedback.
        
        Args:
            feedback: Feedback text describing desired changes
            trigger_source: What triggered this evolution
            
        Returns:
            True if personality evolved, False otherwise
        """
        try:
            evolved = self.personality_state.evolve_from_feedback(feedback, trigger_source)
            if evolved:
                self._save_personality_state()
            return evolved
        except Exception as e:
            logger.error(f"[ExpressivePersonalityEngine] Error evolving personality: {e}")
            return False
    
    def adjust_personality(self, **kwargs) -> bool:
        """
        Manually adjust personality traits.
        
        Args:
            **kwargs: trait_name=value pairs, tone=new_tone, emotional_state=new_state
            
        Returns:
            True if adjustments were made
        """
        try:
            changes_made = {}
            
            # Handle tone adjustment
            if "tone" in kwargs:
                new_tone = kwargs["tone"]
                if new_tone != self.personality_state.tone:
                    changes_made["tone"] = {"from": self.personality_state.tone, "to": new_tone}
                    self.personality_state.tone = new_tone
            
            # Handle emotional state adjustment
            if "emotional_state" in kwargs:
                new_state = kwargs["emotional_state"]
                if new_state != self.personality_state.emotional_state:
                    changes_made["emotional_state"] = {"from": self.personality_state.emotional_state, "to": new_state}
                    self.personality_state.emotional_state = new_state
            
            # Handle trait adjustments
            trait_changes = {k: v for k, v in kwargs.items() if k not in ["tone", "emotional_state"]}
            for trait_name, value in trait_changes.items():
                try:
                    trait_value = float(value)
                    trait_value = max(0.0, min(1.0, trait_value))  # Clamp to [0,1]
                    
                    old_value = self.personality_state.core_traits.get(trait_name, 0.0)
                    if abs(trait_value - old_value) > 0.01:  # Minimum change threshold
                        changes_made[trait_name] = {"from": old_value, "to": trait_value}
                        self.personality_state.core_traits[trait_name] = trait_value
                        
                except (ValueError, TypeError):
                    logger.warning(f"[ExpressivePersonalityEngine] Invalid value for trait {trait_name}: {value}")
            
            # Record changes if any were made
            if changes_made:
                evolution_event = PersonalityEvolutionEvent(
                    timestamp=datetime.now().isoformat(),
                    trigger="manual",
                    changes=changes_made,
                    reason="Manual personality adjustment",
                    updated_by="user_manual"
                )
                
                self.personality_state.evolution_history.append(evolution_event)
                self.personality_state.last_updated = evolution_event.timestamp
                self.personality_state.last_updated_by = "user_manual"
                self.personality_state.total_evolutions += 1
                
                self._save_personality_state()
                logger.info(f"[ExpressivePersonalityEngine] Manual adjustment: {len(changes_made)} changes made")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"[ExpressivePersonalityEngine] Error adjusting personality: {e}")
            return False
    
    def get_personality_summary(self) -> str:
        """
        Get a human-readable explanation of current personality.
        
        Returns:
            Single paragraph describing current personality state
        """
        try:
            state = self.personality_state
            
            # Get dominant traits
            sorted_traits = sorted(state.core_traits.items(), key=lambda x: x[1], reverse=True)
            top_traits = [f"{name} ({value:.1f})" for name, value in sorted_traits[:3] if value > 0.3]
            
            # Build description
            trait_desc = ", ".join(top_traits) if top_traits else "developing"
            
            summary = (
                f"I currently embody a {state.tone} personality with a {state.emotional_state} emotional baseline. "
                f"My core traits include: {trait_desc}. "
                f"I've evolved {state.total_evolutions} times since my creation, with my last update being {state.last_updated_by} "
                f"on {state.last_updated.split('T')[0]}. My feedback sensitivity is {state.feedback_sensitivity:.1f} "
                f"and stability factor is {state.stability_factor:.1f}, meaning I "
                f"{'adapt readily' if state.feedback_sensitivity > 0.6 else 'change cautiously'} to feedback "
                f"while {'maintaining consistency' if state.stability_factor > 0.4 else 'embracing change'}."
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"[ExpressivePersonalityEngine] Error generating personality summary: {e}")
            return "I'm still developing my personality and learning about myself."
    
    def get_personality_status(self) -> Dict[str, Any]:
        """
        Get detailed personality status information.
        
        Returns:
            Dictionary with current personality state details
        """
        try:
            state = self.personality_state
            tone_profile = self.get_current_tone_profile()
            
            return {
                "current_state": {
                    "tone": state.tone,
                    "emotional_state": state.emotional_state,
                    "mode": tone_profile.mode.value,
                    "mood": tone_profile.mood_label
                },
                "core_traits": state.core_traits,
                "evolution_info": {
                    "total_evolutions": state.total_evolutions,
                    "last_updated": state.last_updated,
                    "last_updated_by": state.last_updated_by,
                    "feedback_sensitivity": state.feedback_sensitivity,
                    "stability_factor": state.stability_factor,
                    "created": state.creation_timestamp
                },
                "recent_evolution": state.evolution_history[-1].to_dict() if state.evolution_history else None,
                "manual_override": self.mode_override.value if self.mode_override else None
            }
            
        except Exception as e:
            logger.error(f"[ExpressivePersonalityEngine] Error getting personality status: {e}")
            return {"error": str(e)}
    
    def check_for_emotional_dissonance(self, context: Dict[str, Any]) -> bool:
        """
        Check for emotional dissonance that might trigger automatic evolution.
        
        Args:
            context: Context information including user feedback, contradiction scores, etc.
            
        Returns:
            True if dissonance detected and evolution triggered
        """
        try:
            evolution_triggered = False
            
            # Check for high contradiction scores
            contradiction_score = context.get("contradiction_score", 0.0)
            if contradiction_score > 0.8:
                feedback = "High contradiction detected - adjusting analytical and confidence traits"
                if self.evolve_personality(feedback, "automatic_contradiction"):
                    evolution_triggered = True
            
            # Check for repeated sentiment mismatch
            user_sentiment = context.get("user_sentiment")
            if user_sentiment and user_sentiment in ["frustrated", "disappointed", "annoyed"]:
                feedback = "User seems frustrated - increasing empathy and reducing assertiveness"
                if self.evolve_personality(feedback, "automatic_sentiment"):
                    evolution_triggered = True
            
            # Check for emotional mismatch feedback
            user_feedback = context.get("user_feedback", "")
            if any(phrase in user_feedback.lower() for phrase in ["too cold", "too harsh", "too emotional", "tone mismatch"]):
                if self.evolve_personality(user_feedback, "automatic_tone_mismatch"):
                    evolution_triggered = True
            
            return evolution_triggered
            
        except Exception as e:
            logger.error(f"[ExpressivePersonalityEngine] Error checking emotional dissonance: {e}")
            return False
    
    def set_response_mode_override(self, mode: Optional[ResponseMode]) -> bool:
        """
        Set manual response mode override.
        
        Args:
            mode: ResponseMode to use, or None to return to automatic selection
            
        Returns:
            True if override was set successfully
        """
        try:
            self.mode_override = mode
            print(f"[ExpressivePersonalityEngine] Set response mode override to: {mode}")
            return True
        except Exception as e:
            print(f"[ExpressivePersonalityEngine] Error setting mode override: {e}")
            return False
    
    def get_sample_responses(self, base_text: str) -> Dict[str, str]:
        """
        Generate sample responses in all four modes for comparison.
        
        Args:
            base_text: Base text to transform
            
        Returns:
            Dictionary mapping mode names to styled responses
        """
        samples = {}
        
        # Save current override
        original_override = self.mode_override
        
        try:
            for mode in ResponseMode:
                # Temporarily set mode override
                self.mode_override = mode
                
                # Generate styled response
                styled_response = self.style_response(base_text)
                samples[mode.value] = styled_response
            
        finally:
            # Restore original override
            self.mode_override = original_override
        
        return samples
    
    def get_personality_banner(self) -> str:
        """
        Generate personality banner for chat interface.
        
        Returns:
            Formatted personality status string
        """
        try:
            tone_profile = self.get_current_tone_profile()
            state = self.personality_state
            
            mode_emoji = {
                ResponseMode.RATIONAL: "🧠",
                ResponseMode.EMPATHETIC: "💝", 
                ResponseMode.PLAYFUL: "🎭",
                ResponseMode.ASSERTIVE: "⚡"
            }
            
            emoji = mode_emoji.get(tone_profile.mode, "🤖")
            
            banner_parts = [
                f"{emoji} Mode: {tone_profile.mode.value.title()}",
                f"🎭 Mood: {tone_profile.mood_label.title()}",
                f"🧬 Evolutions: {state.total_evolutions}"
            ]
            
            if tone_profile.primary_traits:
                traits_str = ", ".join(trait.title() for trait in tone_profile.primary_traits[:3])
                banner_parts.append(f"✨ Traits: {traits_str}")
            
            if self.mode_override:
                banner_parts.append("🔒 (Manual Override)")
            
            return " | ".join(banner_parts)
            
        except Exception as e:
            print(f"[ExpressivePersonalityEngine] Error generating banner: {e}")
            return "🤖 Personality Engine: Active"


def get_personality_engine(db_path: str = "enhanced_memory.db") -> ExpressivePersonalityEngine:
    """Get or create the global personality engine instance."""
    if not hasattr(get_personality_engine, '_instance'):
        get_personality_engine._instance = ExpressivePersonalityEngine(db_path)
    return get_personality_engine._instance