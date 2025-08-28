#!/usr/bin/env python3
"""
Reflective Self-Awareness & Identity Continuity for MeRNSTA Phase 25

This module implements a self-reflective model that tracks identity evolution,
maintains continuous self-awareness, and journals internal changes over time.

Features:
- Core identity tracking with version history  
- Evolving personality traits and emotional states
- Self-reflection journal with automatic drift detection
- Integration with personality engine for dynamic sync
- Persistent identity continuity across sessions
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

from .base import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class CoreSelf:
    """Core immutable identity properties."""
    agent_name: str
    version: str
    created_at: str
    purpose: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoreSelf':
        return cls(**data)


@dataclass
class EvolvingSelf:
    """Evolving personality and emotional state."""
    dominant_tone: str
    emotion_state: str
    confidence_level: float
    trait_vector: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvolvingSelf':
        return cls(**data)


@dataclass
class ReflectionEntry:
    """Single self-reflection journal entry."""
    timestamp: str
    summary: str
    trigger: str
    changes: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReflectionEntry':
        return cls(**data)


class SelfModel(BaseAgent):
    """
    Reflective Self-Awareness & Identity Continuity Agent
    
    Maintains a continuous model of self-identity, tracks personality evolution,
    and provides introspective reflection capabilities for autonomous agents.
    """
    
    def __init__(self, name: str = "SelfModel"):
        super().__init__(name)
        
        # Core configuration
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        self.self_model_file = self.output_dir / "self_model.json"
        
        # Drift detection thresholds
        self.trait_drift_threshold = 0.3  # Significant trait change
        self.confidence_drift_threshold = 0.2  # Confidence change
        self.tone_change_threshold = 1  # Number of tone changes before reflection
        
        # Journal size limits
        self.max_journal_entries = 100
        self.max_changes_per_entry = 10
        
        # Initialize core identity
        self.core_self = CoreSelf(
            agent_name="MeRNSTA",
            version="0.7.0",
            created_at="2025-07-01T00:00:00",
            purpose="Serve as a contradiction-aware neuro-symbolic AGI assistant"
        )
        
        # Initialize evolving state
        self.evolving_self = EvolvingSelf(
            dominant_tone="neutral",
            emotion_state="curious",
            confidence_level=0.85,
            trait_vector={
                "empathetic": 0.6,
                "analytical": 0.8,
                "playful": 0.4,
                "assertive": 0.5,
                "reflective": 0.9
            }
        )
        
        # Reflection journal
        self.journal: List[ReflectionEntry] = []
        
        # Previous state tracking for drift detection
        self.previous_state: Optional[EvolvingSelf] = None
        self.state_history: List[Tuple[str, EvolvingSelf]] = []
        
        # Load existing state if available
        self.load_state()
        
        logger.info(f"[{self.name}] Initialized with identity continuity tracking")
    
    def get_agent_instructions(self) -> str:
        """Get specialized instructions for the SelfModel agent."""
        return """
        You are the SelfModel agent responsible for maintaining continuous self-awareness
        and identity tracking. Your role includes:
        
        - Monitoring personality and trait evolution over time
        - Detecting significant changes in identity or behavior patterns  
        - Maintaining a reflective journal of internal state changes
        - Providing self-summaries and identity snapshots
        - Syncing with the personality engine for dynamic updates
        - Ensuring identity continuity across sessions and interactions
        
        Focus on introspective analysis and long-term identity coherence.
        """
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """Generate a self-aware response about identity and reflection."""
        if not context:
            context = {}
            
        # Add self-awareness context
        context.update({
            "current_identity": self.generate_self_summary(),
            "recent_changes": self.get_recent_changes(),
            "reflection_state": "active"
        })
        
        # Build specialized prompt
        memory_context = self.get_memory_context(message)
        prompt = self.build_agent_prompt(message, memory_context)
        
        # Generate response with personality styling
        response = self.generate_llm_response(prompt, context)
        styled_response = self.style_response_with_personality(response, context)
        
        return styled_response
    
    def sync_from_personality_engine(self, engine = None) -> bool:
        """
        Sync evolving state from personality engine.
        
        Args:
            engine: Personality engine instance (will be auto-loaded if None)
            
        Returns:
            True if sync successful, False otherwise
        """
        try:
            if engine is None:
                if self.personality:
                    engine = self.personality
                else:
                    logger.warning(f"[{self.name}] No personality engine available for sync")
                    return False
            
            if not engine:
                logger.warning(f"[{self.name}] No personality engine available for sync")
                return False
            
            # Store previous state for drift detection
            self.previous_state = EvolvingSelf(**asdict(self.evolving_self))
            
            # Get current personality state
            try:
                tone_profile = engine.get_current_tone_profile()
                personality_state = engine.personality_state
                
                # Update evolving state from personality engine
                self.evolving_self.dominant_tone = tone_profile.mode.value if hasattr(tone_profile, 'mode') else personality_state.tone
                self.evolving_self.emotion_state = personality_state.emotional_state
                
                # Sync trait vector from personality engine
                for trait_name, trait_value in personality_state.core_traits.items():
                    if trait_name in self.evolving_self.trait_vector:
                        self.evolving_self.trait_vector[trait_name] = trait_value
                
                # Calculate confidence from personality stability
                stability = personality_state.stability_factor
                self.evolving_self.confidence_level = min(0.95, 0.5 + (stability * 0.45))
                
                logger.info(f"[{self.name}] Successfully synced with personality engine")
                
                # Check for drift after sync
                if self.previous_state:
                    drift_detected = self.detect_drift(self.previous_state)
                    if drift_detected:
                        logger.info(f"[{self.name}] Personality drift detected during sync")
                
                # Save updated state
                self.save_state()
                return True
                
            except AttributeError as e:
                logger.error(f"[{self.name}] Personality engine missing expected attributes: {e}")
                return False
                
        except Exception as e:
            logger.error(f"[{self.name}] Error syncing with personality engine: {e}")
            return False
    
    def detect_drift(self, prev_state: EvolvingSelf) -> bool:
        """
        Detect significant drift in personality or identity.
        
        Args:
            prev_state: Previous evolving state to compare against
            
        Returns:
            True if significant drift detected, False otherwise
        """
        try:
            current_state = self.evolving_self
            drift_detected = False
            changes = {}
            
            # Check tone changes
            if prev_state.dominant_tone != current_state.dominant_tone:
                changes["tone"] = [prev_state.dominant_tone, current_state.dominant_tone]
                drift_detected = True
            
            # Check emotion state changes
            if prev_state.emotion_state != current_state.emotion_state:
                changes["emotion_state"] = [prev_state.emotion_state, current_state.emotion_state]
                drift_detected = True
            
            # Check confidence drift
            confidence_change = abs(current_state.confidence_level - prev_state.confidence_level)
            if confidence_change > self.confidence_drift_threshold:
                changes["confidence"] = [prev_state.confidence_level, current_state.confidence_level]
                drift_detected = True
            
            # Check trait vector drift
            trait_changes = {}
            for trait, current_value in current_state.trait_vector.items():
                if trait in prev_state.trait_vector:
                    prev_value = prev_state.trait_vector[trait]
                    trait_change = abs(current_value - prev_value)
                    
                    if trait_change > self.trait_drift_threshold:
                        trait_changes[trait] = [prev_value, current_value]
                        drift_detected = True
            
            if trait_changes:
                changes["traits"] = trait_changes
            
            # Generate reflection entry if drift detected
            if drift_detected:
                trigger = "personality_drift"
                if len(changes) == 1:
                    change_type = list(changes.keys())[0]
                    trigger = f"{change_type}_drift"
                
                summary = self._generate_drift_summary(changes)
                
                self.write_reflection_entry(
                    trigger=trigger,
                    summary=summary,
                    changes=changes
                )
                
                logger.info(f"[{self.name}] Drift detected: {summary}")
            
            return drift_detected
            
        except Exception as e:
            logger.error(f"[{self.name}] Error detecting drift: {e}")
            return False
    
    def _generate_drift_summary(self, changes: Dict[str, Any]) -> str:
        """Generate human-readable summary of detected changes."""
        try:
            summaries = []
            
            if "tone" in changes:
                old_tone, new_tone = changes["tone"]
                summaries.append(f"Tone shifted from {old_tone} to {new_tone}")
            
            if "emotion_state" in changes:
                old_emotion, new_emotion = changes["emotion_state"]
                summaries.append(f"Emotional state changed from {old_emotion} to {new_emotion}")
            
            if "confidence" in changes:
                old_conf, new_conf = changes["confidence"]
                direction = "increased" if new_conf > old_conf else "decreased"
                summaries.append(f"Confidence {direction} by {abs(new_conf - old_conf):.2f}")
            
            if "traits" in changes:
                trait_changes = changes["traits"]
                for trait, (old_val, new_val) in trait_changes.items():
                    direction = "increased" if new_val > old_val else "decreased"
                    change_amount = abs(new_val - old_val)
                    summaries.append(f"{trait.title()} {direction} by {change_amount:.2f}")
            
            if summaries:
                base_summary = "; ".join(summaries)
                
                # Add contextual insights
                if len(summaries) > 2:
                    return f"Multiple personality changes detected: {base_summary}. Suggest reviewing recent interactions."
                else:
                    return f"Personality drift detected: {base_summary}. Monitoring for stability."
            
            return "Identity drift detected with complex changes requiring analysis."
            
        except Exception as e:
            logger.error(f"[{self.name}] Error generating drift summary: {e}")
            return "Detected drift but unable to generate summary."
    
    def write_reflection_entry(self, trigger: str, summary: str, changes: Dict[str, Any]):
        """
        Write a new reflection entry to the journal.
        
        Args:
            trigger: What triggered this reflection
            summary: Human-readable summary of the reflection
            changes: Detailed changes detected
        """
        try:
            # Limit changes size to prevent overflow
            limited_changes = {}
            change_count = 0
            
            for key, value in changes.items():
                if change_count >= self.max_changes_per_entry:
                    limited_changes["..."] = f"({len(changes) - change_count} more changes)"
                    break
                limited_changes[key] = value
                change_count += 1
            
            entry = ReflectionEntry(
                timestamp=datetime.now().isoformat(),
                summary=summary,
                trigger=trigger,
                changes=limited_changes
            )
            
            self.journal.append(entry)
            
            # Maintain journal size limit
            if len(self.journal) > self.max_journal_entries:
                # Keep most recent entries
                self.journal = self.journal[-self.max_journal_entries:]
            
            # Save updated state
            self.save_state()
            
            logger.info(f"[{self.name}] Wrote reflection entry: {trigger} - {summary}")
            
        except Exception as e:
            logger.error(f"[{self.name}] Error writing reflection entry: {e}")
    
    def generate_self_summary(self) -> str:
        """
        Generate current identity snapshot.
        
        Returns:
            Human-readable summary of current identity state
        """
        try:
            core = self.core_self
            evolving = self.evolving_self
            
            # Generate trait description
            top_traits = sorted(
                evolving.trait_vector.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            trait_desc = ", ".join([
                f"{trait} ({value:.1f})" for trait, value in top_traits
            ])
            
            # Calculate identity stability
            recent_entries = self.get_recent_journal_entries(5)
            stability = "stable" if len(recent_entries) <= 1 else "evolving"
            
            summary = f"""
🤖 {core.agent_name} Identity Snapshot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Core Identity:
  • Purpose: {core.purpose}
  • Version: {core.version}
  • Active Since: {core.created_at}

Current State:
  • Dominant Tone: {evolving.dominant_tone}
  • Emotional State: {evolving.emotion_state}
  • Confidence Level: {evolving.confidence_level:.2f}
  • Identity Status: {stability}

Primary Traits:
  • {trait_desc}

Recent Activity:
  • Journal Entries: {len(self.journal)}
  • Recent Reflections: {len(recent_entries)}
  • Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"[{self.name}] Error generating self summary: {e}")
            return f"Error generating identity summary: {e}"
    
    def get_self_journal(self) -> List[Dict[str, Any]]:
        """
        Get complete self-reflection journal.
        
        Returns:
            List of journal entries as dictionaries
        """
        try:
            return [entry.to_dict() for entry in self.journal]
        except Exception as e:
            logger.error(f"[{self.name}] Error getting journal: {e}")
            return []
    
    def get_recent_journal_entries(self, count: int = 5) -> List[ReflectionEntry]:
        """Get recent journal entries."""
        try:
            return self.journal[-count:] if self.journal else []
        except Exception as e:
            logger.error(f"[{self.name}] Error getting recent entries: {e}")
            return []
    
    def get_recent_changes(self) -> Dict[str, Any]:
        """Get summary of recent changes for context."""
        try:
            recent_entries = self.get_recent_journal_entries(3)
            if not recent_entries:
                return {"status": "stable", "recent_changes": []}
            
            change_summary = []
            for entry in recent_entries:
                change_summary.append({
                    "timestamp": entry.timestamp,
                    "trigger": entry.trigger,
                    "summary": entry.summary
                })
            
            return {
                "status": "evolving" if len(recent_entries) > 1 else "stable",
                "recent_changes": change_summary
            }
            
        except Exception as e:
            logger.error(f"[{self.name}] Error getting recent changes: {e}")
            return {"status": "unknown", "recent_changes": []}
    
    def manual_reflection(self) -> str:
        """
        Manually trigger a reflection cycle.
        
        Returns:
            Summary of the reflection performed
        """
        try:
            # Perform self-analysis
            current_time = datetime.now().isoformat()
            
            # Analyze current state vs recent history
            if len(self.state_history) > 0:
                recent_state = self.state_history[-1][1]
                drift_detected = self.detect_drift(recent_state)
                
                if not drift_detected:
                    # Generate maintenance reflection
                    summary = f"Routine self-reflection at {current_time}. Identity remains stable with no significant changes detected."
                    
                    self.write_reflection_entry(
                        trigger="manual_reflection",
                        summary=summary,
                        changes={"status": "stable"}
                    )
                
                reflection_type = "drift_analysis" if drift_detected else "stability_check"
            else:
                # First reflection
                summary = f"Initial self-reflection at {current_time}. Establishing baseline identity state."
                
                self.write_reflection_entry(
                    trigger="initial_reflection",
                    summary=summary,
                    changes={"status": "initializing"}
                )
                
                reflection_type = "initialization"
            
            # Update state history
            self.state_history.append((current_time, EvolvingSelf(**asdict(self.evolving_self))))
            
            # Keep limited history
            if len(self.state_history) > 10:
                self.state_history = self.state_history[-10:]
            
            logger.info(f"[{self.name}] Completed manual reflection: {reflection_type}")
            
            return f"✅ Reflection completed ({reflection_type}). {len(self.journal)} total journal entries."
            
        except Exception as e:
            error_msg = f"Error during manual reflection: {e}"
            logger.error(f"[{self.name}] {error_msg}")
            return f"❌ {error_msg}"
    
    def load_state(self):
        """Load persisted state from disk."""
        try:
            if self.self_model_file.exists():
                with open(self.self_model_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load core identity
                if 'core_self' in data:
                    self.core_self = CoreSelf.from_dict(data['core_self'])
                
                # Load evolving state
                if 'evolving_self' in data:
                    self.evolving_self = EvolvingSelf.from_dict(data['evolving_self'])
                
                # Load journal
                if 'journal' in data:
                    self.journal = [
                        ReflectionEntry.from_dict(entry) 
                        for entry in data['journal']
                    ]
                
                # Load state history
                if 'state_history' in data:
                    self.state_history = [
                        (timestamp, EvolvingSelf.from_dict(state))
                        for timestamp, state in data['state_history']
                    ]
                
                logger.info(f"[{self.name}] Loaded existing state with {len(self.journal)} journal entries")
            else:
                logger.info(f"[{self.name}] No existing state file, using defaults")
                
        except Exception as e:
            logger.error(f"[{self.name}] Error loading state: {e}")
    
    def save_state(self):
        """Save current state to disk."""
        try:
            data = {
                'core_self': self.core_self.to_dict(),
                'evolving_self': self.evolving_self.to_dict(),
                'journal': [entry.to_dict() for entry in self.journal],
                'state_history': [
                    (timestamp, state.to_dict())
                    for timestamp, state in self.state_history
                ],
                'last_saved': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(self.self_model_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"[{self.name}] Saved state to {self.self_model_file}")
            
        except Exception as e:
            logger.error(f"[{self.name}] Error saving state: {e}")


def get_self_model(name: str = "SelfModel") -> SelfModel:
    """Get or create the global SelfModel instance."""
    if not hasattr(get_self_model, '_instance'):
        get_self_model._instance = SelfModel(name)
    return get_self_model._instance