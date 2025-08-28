#!/usr/bin/env python3
"""
PersonalityEvolver for MeRNSTA Phase 29
    
Memory-driven personality shifts that enable dynamic, long-term evolution of the agent's 
tone, traits, and behavioral tendencies by learning from memory trends, emotional volatility, 
and internal contradictions.
"""

import json
import logging
import time
import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np

from agents.base import BaseAgent
from config.settings import get_config

logger = logging.getLogger(__name__)


@dataclass
class PersonalityTraitChange:
    """Represents a change in a specific personality trait."""
    trait_name: str
    old_value: float
    new_value: float
    change_magnitude: float
    trigger_type: str  # "memory_trend", "contradiction_stress", "emotional_drift", "belief_change"
    trigger_details: str
    confidence: float
    timestamp: float
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class PersonalityEvolutionTrace:
    """Comprehensive trace of a personality evolution event."""
    evolution_id: str
    trigger_type: str  # "weekly_cadence", "major_conflict", "stress_threshold", "manual"
    timestamp: float
    
    # Analysis data
    memory_analysis: Dict[str, Any]
    contradiction_analysis: Dict[str, Any]
    emotional_analysis: Dict[str, Any]
    dissonance_pressure: float
    
    # Changes made
    trait_changes: List[PersonalityTraitChange]
    tone_changes: Dict[str, Any]
    mood_changes: Dict[str, Any]
    
    # Evolution rationale
    justification: str
    natural_language_summary: str
    
    # Metadata
    duration_ms: float
    total_changes: int
    evolution_magnitude: float  # 0-1 scale of how significant this evolution was
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PersonalityVector:
    """Current personality state vector with trait weights and configurations."""
    # Core traits (0.0 to 1.0 scale)
    curiosity: float = 0.5
    caution: float = 0.5
    empathy: float = 0.5
    assertiveness: float = 0.5
    optimism: float = 0.5
    analytical: float = 0.5
    creativity: float = 0.5
    confidence: float = 0.5
    skepticism: float = 0.5
    emotional_sensitivity: float = 0.5
    
    # Tone configuration
    base_tone: str = "balanced"  # "analytical", "empathetic", "assertive", "playful", "balanced"
    tone_variance: float = 0.3  # How much tone can vary from base (0-1)
    
    # Emotional baseline
    emotional_baseline: str = "stable"  # "stable", "volatile", "optimistic", "cautious"
    emotional_reactivity: float = 0.5  # How reactive to emotional stimuli (0-1)
    
    # Evolution metadata
    last_evolution: float = 0.0
    evolution_count: int = 0
    total_evolution_magnitude: float = 0.0
    
    def __post_init__(self):
        if self.last_evolution == 0.0:
            self.last_evolution = time.time()
    
    def get_trait_dict(self) -> Dict[str, float]:
        """Get all traits as a dictionary."""
        return {
            'curiosity': self.curiosity,
            'caution': self.caution,
            'empathy': self.empathy,
            'assertiveness': self.assertiveness,
            'optimism': self.optimism,
            'analytical': self.analytical,
            'creativity': self.creativity,
            'confidence': self.confidence,
            'skepticism': self.skepticism,
            'emotional_sensitivity': self.emotional_sensitivity
        }
    
    def update_trait(self, trait_name: str, new_value: float) -> bool:
        """Update a trait value with bounds checking."""
        if trait_name not in self.get_trait_dict():
            logger.warning(f"Unknown trait: {trait_name}")
            return False
            
        # Apply bounds (0.0 to 1.0)
        bounded_value = max(0.0, min(1.0, new_value))
        setattr(self, trait_name, bounded_value)
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PersonalityEvolver(BaseAgent):
    """
    Analyzes memory entries, contradiction logs, and dissonance pressure to evolve
    personality traits over time. Supports short-term (mood), mid-term (style), 
    and long-term (trait) adjustments.
    """
    
    def __init__(self):
        super().__init__("personality_evolver")
        
        # Load configuration
        self.config = get_config()
        self.evolution_config = self.config.get('personality_evolution', {})
        
        # Configurable parameters [[memory:4199483]]
        self.sensitivity_threshold = self.evolution_config.get('sensitivity_threshold', 0.3)
        self.max_shift_rate = self.evolution_config.get('max_shift_rate', 0.2)
        self.mood_decay_rate = self.evolution_config.get('mood_decay_rate', 0.1)
        self.trait_bounds = self.evolution_config.get('trait_bounds', {'min': 0.05, 'max': 0.95})
        self.weekly_cadence_hours = self.evolution_config.get('weekly_cadence_hours', 168)  # 1 week
        self.major_conflict_threshold = self.evolution_config.get('major_conflict_threshold', 0.8)
        
        # Evolution toggles
        self.enable_short_term = self.evolution_config.get('enable_short_term_evolution', True)
        self.enable_long_term = self.evolution_config.get('enable_long_term_evolution', True)
        self.enable_automatic_triggers = self.evolution_config.get('enable_automatic_triggers', True)
        
        # State
        self.personality_vector = PersonalityVector()
        self.evolution_history: List[PersonalityEvolutionTrace] = []
        self.last_weekly_check = time.time()
        
        # Storage paths
        self.state_file = Path("output/personality_state.json")
        self.history_file = Path("output/personality_evolution.jsonl")
        self.state_file.parent.mkdir(exist_ok=True)
        
        # Load persistent state
        self._load_persistent_state()
        
        # Lazy-loaded components
        self._dissonance_tracker = None
        self._enhanced_memory = None
        
        logger.info(f"[{self.name}] Initialized with sensitivity_threshold={self.sensitivity_threshold}")
    
    def get_agent_instructions(self) -> str:
        """Get instructions for the PersonalityEvolver agent."""
        return """You are the PersonalityEvolver, responsible for memory-driven personality shifts.

Your primary functions:
1. Analyze memory trends, emotional patterns, and contradiction stress
2. Evolve personality traits based on accumulated experiences and beliefs
3. Maintain personality continuity while adapting to cognitive pressures
4. Trigger evolution on weekly cadence and major belief conflicts
5. Provide explainable personality changes with natural language justification

You work autonomously during meta-reflection cycles but can also be manually triggered."""
    
    def respond(self, message: str, context: Dict[str, Any] = None) -> str:
        """
        Respond to queries about personality evolution.
        
        Args:
            message: User message or query
            context: Additional context for the response
            
        Returns:
            Response about personality evolution status or capabilities
        """
        try:
            if not message:
                return self._generate_status_response()
            
            message_lower = message.lower()
            
            if any(word in message_lower for word in ['status', 'current', 'traits']):
                return self._generate_status_response()
            
            elif any(word in message_lower for word in ['evolve', 'change', 'adapt']):
                return self._generate_evolution_response()
            
            elif any(word in message_lower for word in ['history', 'past', 'changes']):
                return self._generate_history_response()
            
            else:
                return self._generate_general_response()
        
        except Exception as e:
            logger.error(f"[{self.name}] Error in respond: {e}")
            return f"I encountered an issue while processing your personality evolution query: {str(e)}"
    
    def _generate_status_response(self) -> str:
        """Generate a status response about current personality traits."""
        try:
            status = self.get_personality_status()
            traits = status['current_traits']
            
            # Find most and least prominent traits
            sorted_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)
            highest_traits = sorted_traits[:3]
            lowest_traits = sorted_traits[-3:]
            
            response = f"🧬 **Current Personality Profile**\n\n"
            response += f"**Strongest Traits:**\n"
            for trait, value in highest_traits:
                response += f"• {trait.title()}: {value:.2f}\n"
            
            response += f"\n**Areas for Growth:**\n"
            for trait, value in lowest_traits:
                response += f"• {trait.title()}: {value:.2f}\n"
            
            meta = status['evolution_metadata']
            response += f"\n**Evolution Summary:**\n"
            response += f"• Total Evolutions: {meta['total_evolutions']}\n"
            response += f"• Evolution Magnitude: {meta['total_evolution_magnitude']:.3f}\n"
            
            if status['recent_evolution_history']:
                recent = status['recent_evolution_history'][0]
                response += f"• Recent Change: {recent['summary']}\n"
            
            return response
            
        except Exception as e:
            return f"Error generating personality status: {e}"
    
    def _generate_evolution_response(self) -> str:
        """Generate a response about triggering personality evolution."""
        return """🔄 **Personality Evolution Capabilities**

I can adapt personality traits based on:
• **Memory Trends**: Emotional language patterns and recurring themes
• **Contradiction Stress**: Belief conflicts and cognitive dissonance
• **Behavioral Feedback**: Success/failure patterns and goal achievement

Evolution triggers:
• Weekly automatic checks during meta-reflection
• Major belief conflicts (high dissonance pressure)
• Manual triggers via `/personality_evolve` command

All changes are:
✓ Explainable with natural language justification
✓ Bounded to preserve core identity
✓ Logged for analysis and potential rollback
✓ Integrated with the existing personality system"""
    
    def _generate_history_response(self) -> str:
        """Generate a response about personality evolution history."""
        try:
            history = self.get_evolution_history(limit=3)
            
            if not history:
                return "📚 **Evolution History**\n\nNo personality evolutions have occurred yet. Evolution will happen automatically during weekly meta-reflection cycles or when major belief conflicts arise."
            
            response = f"📚 **Recent Evolution History** (last {len(history)} events)\n\n"
            
            for i, trace in enumerate(history, 1):
                trigger = trace.get('trigger_type', 'unknown')
                magnitude = trace.get('evolution_magnitude', 0)
                summary = trace.get('natural_language_summary', 'No summary')
                
                response += f"**{i}. {trigger.title()} Evolution**\n"
                response += f"• Impact: {magnitude:.3f}\n"
                response += f"• Summary: {summary}\n\n"
            
            return response
            
        except Exception as e:
            return f"Error retrieving evolution history: {e}"
    
    def _generate_general_response(self) -> str:
        """Generate a general response about personality evolution."""
        return """🧠 **Personality Evolution System**

I'm responsible for dynamic personality adaptation based on your experiences and cognitive patterns. I monitor:

**Memory Analysis**: Emotional language trends, themes, and volatility
**Contradiction Tracking**: Belief conflicts and cognitive dissonance
**Behavioral Patterns**: Success rates and goal achievement

**Key Features**:
• Gradual, explainable personality shifts
• Preserves core identity while adapting traits
• Weekly automatic evolution during meta-reflection
• Emergency evolution during major belief conflicts

Use `/personality_status` to see current traits or `/personality_evolve` to trigger manual evolution."""
    
    @property
    def dissonance_tracker(self):
        """Lazy-load dissonance tracker for contradiction analysis"""
        if self._dissonance_tracker is None:
            try:
                from agents.dissonance_tracker import DissonanceTracker
                self._dissonance_tracker = DissonanceTracker()
            except ImportError as e:
                logger.error(f"[{self.name}] Could not load DissonanceTracker: {e}")
                self._dissonance_tracker = None
        return self._dissonance_tracker
    
    @property 
    def enhanced_memory(self):
        """Lazy-load enhanced memory system for semantic analysis"""
        if self._enhanced_memory is None:
            try:
                from storage.enhanced_memory_system import EnhancedMemorySystem
                self._enhanced_memory = EnhancedMemorySystem()
            except ImportError as e:
                logger.error(f"[{self.name}] Could not load EnhancedMemorySystem: {e}")
                self._enhanced_memory = None
        return self._enhanced_memory
    
    def _load_persistent_state(self):
        """Load personality state from persistent storage."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                
                # Load personality vector
                if 'personality_vector' in state_data:
                    vector_data = state_data['personality_vector']
                    for key, value in vector_data.items():
                        if hasattr(self.personality_vector, key):
                            setattr(self.personality_vector, key, value)
                
                # Load evolution history metadata
                self.last_weekly_check = state_data.get('last_weekly_check', time.time())
                
                logger.info(f"[{self.name}] Loaded personality state with {self.personality_vector.evolution_count} evolutions")
            
            # Load evolution history
            if self.history_file.exists():
                self.evolution_history = []
                with open(self.history_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                trace_data = json.loads(line)
                                # Note: We load as dict rather than full dataclass for performance
                                self.evolution_history.append(trace_data)
                            except json.JSONDecodeError:
                                continue
                
                logger.info(f"[{self.name}] Loaded {len(self.evolution_history)} evolution traces")
                
        except Exception as e:
            logger.error(f"[{self.name}] Error loading persistent state: {e}")
    
    def _save_persistent_state(self):
        """Save personality state to persistent storage."""
        try:
            # Save current state
            state_data = {
                'personality_vector': self.personality_vector.to_dict(),
                'last_weekly_check': self.last_weekly_check,
                'last_updated': time.time(),
                'version': '1.0'
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.debug(f"[{self.name}] Saved personality state")
            
        except Exception as e:
            logger.error(f"[{self.name}] Error saving persistent state: {e}")
    
    def _save_evolution_trace(self, trace: PersonalityEvolutionTrace):
        """Save evolution trace to JSONL history."""
        try:
            with open(self.history_file, 'a') as f:
                f.write(json.dumps(trace.to_dict()) + '\n')
            
            # Also add to in-memory history
            self.evolution_history.append(trace.to_dict())
            
            logger.debug(f"[{self.name}] Saved evolution trace {trace.evolution_id}")
            
        except Exception as e:
            logger.error(f"[{self.name}] Error saving evolution trace: {e}")
    
    def analyze_memory_trends(self, lookback_days: int = 7) -> Dict[str, Any]:
        """
        Analyze memory entries for emotional patterns, themes, and trends.
        
        Args:
            lookback_days: Number of days to look back for analysis
            
        Returns:
            Analysis results with themes, emotional patterns, and trend indicators
        """
        try:
            analysis = {
                'themes': [],
                'emotional_drift': 0.0,
                'volatility_score': 0.0,
                'dominant_emotions': [],
                'memory_volume': 0,
                'fact_categories': {},
                'temporal_patterns': {}
            }
            
            # Get recent memory facts
            if not self.memory_system:
                logger.warning(f"[{self.name}] Memory system not available for trend analysis")
                return analysis
            
            # Calculate lookback timestamp
            lookback_timestamp = time.time() - (lookback_days * 24 * 3600)
            
            # Analyze facts from memory system
            try:
                facts = self.memory_system.get_facts(user_profile_id="personality_analysis")
                recent_facts = [f for f in facts if (f.timestamp or 0) > lookback_timestamp]
                
                analysis['memory_volume'] = len(recent_facts)
                
                if recent_facts:
                    # Analyze themes and subjects
                    subjects = [f.subject.lower() for f in recent_facts]
                    subject_counts = defaultdict(int)
                    for subject in subjects:
                        subject_counts[subject] += 1
                    
                    # Top themes by frequency
                    analysis['themes'] = [
                        {'theme': subj, 'frequency': count} 
                        for subj, count in sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    ]
                    
                    # Analyze emotional language if sentiment data is available
                    emotional_words = 0
                    total_words = 0
                    for fact in recent_facts:
                        fact_text = f"{fact.subject} {fact.predicate} {fact.object}".lower()
                        words = fact_text.split()
                        total_words += len(words)
                        
                        # Simple emotional word detection
                        emotion_indicators = ['feel', 'emotion', 'happy', 'sad', 'angry', 'frustrated', 'excited', 'worried', 'confident', 'uncertain']
                        emotional_words += sum(1 for word in words if any(indicator in word for indicator in emotion_indicators))
                    
                    if total_words > 0:
                        analysis['emotional_drift'] = emotional_words / total_words
                    
                    # Calculate volatility from contradiction scores if available
                    contradiction_scores = [getattr(f, 'contradiction_score', 0.0) for f in recent_facts]
                    if contradiction_scores:
                        analysis['volatility_score'] = statistics.stdev(contradiction_scores) if len(contradiction_scores) > 1 else 0.0
                
            except Exception as e:
                logger.error(f"[{self.name}] Error analyzing memory facts: {e}")
            
            logger.debug(f"[{self.name}] Memory trend analysis: {analysis['memory_volume']} facts, {len(analysis['themes'])} themes")
            return analysis
            
        except Exception as e:
            logger.error(f"[{self.name}] Error in memory trend analysis: {e}")
            return analysis
    
    def analyze_contradiction_stress(self) -> Dict[str, Any]:
        """
        Analyze contradiction logs and dissonance pressure for stress indicators.
        
        Returns:
            Stress analysis with pressure levels, unresolved conflicts, and stress indicators
        """
        try:
            analysis = {
                'pressure_level': 0.0,
                'unresolved_conflicts': 0,
                'stress_indicators': [],
                'contradiction_frequency': 0.0,
                'resolution_success_rate': 0.0,
                'dominant_conflict_areas': []
            }
            
            if not self.dissonance_tracker:
                logger.warning(f"[{self.name}] DissonanceTracker not available for stress analysis")
                return analysis
            
            # Analyze dissonance regions
            try:
                dissonance_regions = getattr(self.dissonance_tracker, 'dissonance_regions', {})
                analysis['unresolved_conflicts'] = len(dissonance_regions)
                
                if dissonance_regions:
                    # Calculate average pressure level
                    pressure_levels = []
                    conflict_areas = []
                    
                    for region_id, region in dissonance_regions.items():
                        if hasattr(region, 'pressure_vector'):
                            pressure_levels.append(getattr(region.pressure_vector, 'urgency', 0.0))
                        if hasattr(region, 'semantic_cluster'):
                            conflict_areas.append(region.semantic_cluster)
                    
                    if pressure_levels:
                        analysis['pressure_level'] = statistics.mean(pressure_levels)
                    
                    # Dominant conflict areas
                    if conflict_areas:
                        area_counts = defaultdict(int)
                        for area in conflict_areas:
                            area_counts[area] += 1
                        analysis['dominant_conflict_areas'] = [
                            {'area': area, 'count': count}
                            for area, count in sorted(area_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                        ]
                
                # Generate stress indicators
                if analysis['pressure_level'] > 0.7:
                    analysis['stress_indicators'].append('high_dissonance_pressure')
                if analysis['unresolved_conflicts'] > 3:
                    analysis['stress_indicators'].append('multiple_unresolved_conflicts')
                
            except Exception as e:
                logger.error(f"[{self.name}] Error analyzing dissonance data: {e}")
            
            logger.debug(f"[{self.name}] Contradiction stress analysis: pressure={analysis['pressure_level']:.2f}, conflicts={analysis['unresolved_conflicts']}")
            return analysis
            
        except Exception as e:
            logger.error(f"[{self.name}] Error in contradiction stress analysis: {e}")
            return analysis
    
    def detect_belief_changes(self, lookback_days: int = 7) -> Dict[str, Any]:
        """
        Detect significant belief changes that might trigger personality evolution.
        
        Args:
            lookback_days: Number of days to analyze for belief changes
            
        Returns:
            Analysis of belief changes with magnitude and areas of change
        """
        try:
            analysis = {
                'belief_changes': [],
                'change_magnitude': 0.0,
                'affected_areas': [],
                'trend_direction': 'stable'  # 'positive', 'negative', 'stable', 'volatile'
            }
            
            # This would integrate with the world_beliefs system when available
            # For now, we approximate through memory analysis
            memory_analysis = self.analyze_memory_trends(lookback_days)
            
            # Detect belief instability through memory volatility
            if memory_analysis['volatility_score'] > 0.5:
                analysis['trend_direction'] = 'volatile'
                analysis['change_magnitude'] = memory_analysis['volatility_score']
                analysis['affected_areas'] = [theme['theme'] for theme in memory_analysis['themes'][:3]]
            
            elif memory_analysis['emotional_drift'] > 0.3:
                analysis['trend_direction'] = 'positive' if memory_analysis['emotional_drift'] > 0.5 else 'negative'
                analysis['change_magnitude'] = abs(memory_analysis['emotional_drift'] - 0.5) * 2
            
            logger.debug(f"[{self.name}] Belief change analysis: magnitude={analysis['change_magnitude']:.2f}, direction={analysis['trend_direction']}")
            return analysis
            
        except Exception as e:
            logger.error(f"[{self.name}] Error in belief change detection: {e}")
            return analysis
    
    def calculate_evolution_pressure(self, memory_analysis: Dict[str, Any], 
                                   contradiction_analysis: Dict[str, Any], 
                                   belief_analysis: Dict[str, Any]) -> float:
        """
        Calculate the overall pressure for personality evolution based on multiple factors.
        
        Args:
            memory_analysis: Results from analyze_memory_trends
            contradiction_analysis: Results from analyze_contradiction_stress  
            belief_analysis: Results from detect_belief_changes
            
        Returns:
            Evolution pressure score (0.0 to 1.0)
        """
        try:
            # Weight different pressure sources
            memory_pressure = memory_analysis.get('volatility_score', 0.0) * 0.3
            contradiction_pressure = contradiction_analysis.get('pressure_level', 0.0) * 0.4
            belief_pressure = belief_analysis.get('change_magnitude', 0.0) * 0.3
            
            total_pressure = memory_pressure + contradiction_pressure + belief_pressure
            
            # Apply non-linear scaling for extreme pressures
            scaled_pressure = 1.0 - math.exp(-total_pressure * 2.0)
            
            logger.debug(f"[{self.name}] Evolution pressure: memory={memory_pressure:.2f}, contradiction={contradiction_pressure:.2f}, belief={belief_pressure:.2f}, total={scaled_pressure:.2f}")
            
            return min(1.0, scaled_pressure)
            
        except Exception as e:
            logger.error(f"[{self.name}] Error calculating evolution pressure: {e}")
            return 0.0
    
    def generate_trait_adjustments(self, memory_analysis: Dict[str, Any],
                                 contradiction_analysis: Dict[str, Any],
                                 belief_analysis: Dict[str, Any],
                                 evolution_pressure: float) -> List[PersonalityTraitChange]:
        """
        Generate specific trait adjustments based on analysis results.
        
        Args:
            memory_analysis: Memory trend analysis
            contradiction_analysis: Contradiction stress analysis
            belief_analysis: Belief change analysis  
            evolution_pressure: Overall evolution pressure score
            
        Returns:
            List of proposed trait changes
        """
        try:
            trait_changes = []
            current_traits = self.personality_vector.get_trait_dict()
            
            # Apply max shift rate constraint [[memory:4199483]]
            max_change = self.max_shift_rate * evolution_pressure
            
            # Skepticism based on contradiction stress
            if contradiction_analysis.get('pressure_level', 0.0) > 0.6:
                current_skepticism = current_traits['skepticism']
                pressure_level = contradiction_analysis['pressure_level']
                new_skepticism = min(1.0, current_skepticism + (pressure_level * max_change))
                
                if abs(new_skepticism - current_skepticism) > 0.05:
                    trait_changes.append(PersonalityTraitChange(
                        trait_name='skepticism',
                        old_value=current_skepticism,
                        new_value=new_skepticism,
                        change_magnitude=abs(new_skepticism - current_skepticism),
                        trigger_type='contradiction_stress',
                        trigger_details=f"High contradiction pressure ({pressure_level:.2f}) increased skepticism",
                        confidence=min(0.9, pressure_level),
                        timestamp=time.time()
                    ))
            
            # Confidence based on resolution success
            resolution_rate = contradiction_analysis.get('resolution_success_rate', 0.5)
            if resolution_rate < 0.4:  # Low success rate
                current_confidence = current_traits['confidence']
                new_confidence = max(0.0, current_confidence - (max_change * 0.5))
                
                if abs(new_confidence - current_confidence) > 0.05:
                    trait_changes.append(PersonalityTraitChange(
                        trait_name='confidence',
                        old_value=current_confidence,
                        new_value=new_confidence,
                        change_magnitude=abs(new_confidence - current_confidence),
                        trigger_type='performance_feedback',
                        trigger_details=f"Low resolution success rate ({resolution_rate:.2f}) decreased confidence",
                        confidence=0.7,
                        timestamp=time.time()
                    ))
            
            # Empathy based on emotional language in memory
            emotional_drift = memory_analysis.get('emotional_drift', 0.0)
            if emotional_drift > 0.2:  # Significant emotional language
                current_empathy = current_traits['empathy']
                new_empathy = min(1.0, current_empathy + (emotional_drift * max_change))
                
                if abs(new_empathy - current_empathy) > 0.05:
                    trait_changes.append(PersonalityTraitChange(
                        trait_name='empathy',
                        old_value=current_empathy,
                        new_value=new_empathy,
                        change_magnitude=abs(new_empathy - current_empathy),
                        trigger_type='emotional_drift',
                        trigger_details=f"Frequent emotional language ({emotional_drift:.2f}) increased empathy",
                        confidence=0.6,
                        timestamp=time.time()
                    ))
            
            # Caution based on belief volatility
            if belief_analysis.get('trend_direction') == 'volatile':
                current_caution = current_traits['caution'] 
                volatility_magnitude = belief_analysis.get('change_magnitude', 0.0)
                new_caution = min(1.0, current_caution + (volatility_magnitude * max_change))
                
                if abs(new_caution - current_caution) > 0.05:
                    trait_changes.append(PersonalityTraitChange(
                        trait_name='caution',
                        old_value=current_caution,
                        new_value=new_caution,
                        change_magnitude=abs(new_caution - current_caution),
                        trigger_type='belief_change',
                        trigger_details=f"Belief volatility ({volatility_magnitude:.2f}) increased caution",
                        confidence=0.8,
                        timestamp=time.time()
                    ))
            
            # Apply trait bounds [[memory:4199483]]
            for change in trait_changes:
                if change.new_value < self.trait_bounds['min']:
                    change.new_value = self.trait_bounds['min']
                elif change.new_value > self.trait_bounds['max']:
                    change.new_value = self.trait_bounds['max']
                
                # Recalculate magnitude after bounds
                change.change_magnitude = abs(change.new_value - change.old_value)
            
            logger.debug(f"[{self.name}] Generated {len(trait_changes)} trait adjustments")
            return trait_changes
            
        except Exception as e:
            logger.error(f"[{self.name}] Error generating trait adjustments: {e}")
            return []
    
    def apply_trait_changes(self, trait_changes: List[PersonalityTraitChange]) -> Dict[str, Any]:
        """
        Apply trait changes to the personality vector.
        
        Args:
            trait_changes: List of trait changes to apply
            
        Returns:
            Summary of changes applied
        """
        try:
            changes_applied = {}
            
            for change in trait_changes:
                if self.personality_vector.update_trait(change.trait_name, change.new_value):
                    changes_applied[change.trait_name] = {
                        'from': change.old_value,
                        'to': change.new_value,
                        'change': change.change_magnitude,
                        'trigger': change.trigger_type
                    }
                    logger.info(f"[{self.name}] Updated {change.trait_name}: {change.old_value:.3f} → {change.new_value:.3f} ({change.trigger_type})")
            
            # Update evolution metadata
            if changes_applied:
                self.personality_vector.evolution_count += 1
                total_magnitude = sum(change.change_magnitude for change in trait_changes)
                self.personality_vector.total_evolution_magnitude += total_magnitude
                self.personality_vector.last_evolution = time.time()
            
            return changes_applied
            
        except Exception as e:
            logger.error(f"[{self.name}] Error applying trait changes: {e}")
            return {}
    
    def generate_natural_language_summary(self, trace: PersonalityEvolutionTrace) -> str:
        """
        Generate a natural language explanation of the personality evolution.
        
        Args:
            trace: Evolution trace to summarize
            
        Returns:
            Human-readable summary of the evolution
        """
        try:
            if not trace.trait_changes:
                return "No significant personality changes detected."
            
            # Categorize changes
            increases = [c for c in trace.trait_changes if c.new_value > c.old_value]
            decreases = [c for c in trace.trait_changes if c.new_value < c.old_value]
            
            summary_parts = []
            
            if increases:
                increased_traits = [f"{c.trait_name} ({c.old_value:.2f}→{c.new_value:.2f})" for c in increases]
                summary_parts.append(f"Increased: {', '.join(increased_traits)}")
            
            if decreases:
                decreased_traits = [f"{c.trait_name} ({c.old_value:.2f}→{c.new_value:.2f})" for c in decreases]
                summary_parts.append(f"Decreased: {', '.join(decreased_traits)}")
            
            # Add context about triggers
            trigger_types = set(c.trigger_type for c in trace.trait_changes)
            if 'contradiction_stress' in trigger_types:
                summary_parts.append("Driven by high contradiction stress and belief conflicts.")
            if 'emotional_drift' in trigger_types:
                summary_parts.append("Responding to increased emotional language patterns.")
            if 'belief_change' in trigger_types:
                summary_parts.append("Adapting to belief volatility and uncertainty.")
            
            # Overall assessment
            if trace.evolution_magnitude > 0.3:
                summary_parts.append("This represents a significant personality shift.")
            elif trace.evolution_magnitude > 0.1:
                summary_parts.append("This represents a moderate personality adjustment.")
            else:
                summary_parts.append("This represents subtle personality refinement.")
            
            return " ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"[{self.name}] Error generating natural language summary: {e}")
            return f"Personality evolution occurred but summary generation failed: {e}"
    
    def evolve_personality(self, trigger_type: str = "manual", 
                         lookback_days: int = 7) -> Optional[PersonalityEvolutionTrace]:
        """
        Perform a complete personality evolution cycle.
        
        Args:
            trigger_type: What triggered this evolution
            lookback_days: How many days of data to analyze
            
        Returns:
            Evolution trace if evolution occurred, None if no changes needed
        """
        try:
            start_time = time.time()
            evolution_id = f"evo_{int(start_time)}"
            
            logger.info(f"[{self.name}] Starting personality evolution {evolution_id} (trigger: {trigger_type})")
            
            # Perform analysis
            memory_analysis = self.analyze_memory_trends(lookback_days)
            contradiction_analysis = self.analyze_contradiction_stress()
            belief_analysis = self.detect_belief_changes(lookback_days)
            
            # Calculate evolution pressure
            evolution_pressure = self.calculate_evolution_pressure(
                memory_analysis, contradiction_analysis, belief_analysis
            )
            
            logger.debug(f"[{self.name}] Evolution pressure: {evolution_pressure:.2f}")
            
            # Check if evolution is warranted
            if evolution_pressure < self.sensitivity_threshold and trigger_type != "manual":
                logger.info(f"[{self.name}] Evolution pressure ({evolution_pressure:.2f}) below threshold ({self.sensitivity_threshold})")
                return None
            
            # Generate trait adjustments
            trait_changes = self.generate_trait_adjustments(
                memory_analysis, contradiction_analysis, belief_analysis, evolution_pressure
            )
            
            if not trait_changes and trigger_type != "manual":
                logger.info(f"[{self.name}] No trait changes needed")
                return None
            
            # Apply changes
            changes_applied = self.apply_trait_changes(trait_changes)
            
            # Create evolution trace
            trace = PersonalityEvolutionTrace(
                evolution_id=evolution_id,
                trigger_type=trigger_type,
                timestamp=start_time,
                memory_analysis=memory_analysis,
                contradiction_analysis=contradiction_analysis,
                emotional_analysis={
                    'emotional_drift': memory_analysis.get('emotional_drift', 0.0),
                    'volatility_score': memory_analysis.get('volatility_score', 0.0)
                },
                dissonance_pressure=contradiction_analysis.get('pressure_level', 0.0),
                trait_changes=trait_changes,
                tone_changes={},  # Future: tone adjustments
                mood_changes={},  # Future: mood adjustments
                justification=f"Evolution triggered by {trigger_type} with pressure {evolution_pressure:.2f}",
                natural_language_summary="",  # Will be filled below
                duration_ms=(time.time() - start_time) * 1000,
                total_changes=len(changes_applied),
                evolution_magnitude=sum(c.change_magnitude for c in trait_changes)
            )
            
            # Generate natural language summary
            trace.natural_language_summary = self.generate_natural_language_summary(trace)
            
            # Save evolution trace
            self._save_evolution_trace(trace)
            self._save_persistent_state()
            
            logger.info(f"[{self.name}] Completed evolution {evolution_id}: {trace.total_changes} changes, magnitude {trace.evolution_magnitude:.3f}")
            logger.info(f"[{self.name}] Summary: {trace.natural_language_summary}")
            
            return trace
            
        except Exception as e:
            logger.error(f"[{self.name}] Error during personality evolution: {e}")
            return None
    
    def check_weekly_cadence(self) -> bool:
        """
        Check if it's time for a weekly personality evolution check.
        
        Returns:
            True if weekly evolution should be triggered
        """
        try:
            current_time = time.time()
            time_since_last = current_time - self.last_weekly_check
            
            if time_since_last >= (self.weekly_cadence_hours * 3600):
                self.last_weekly_check = current_time
                logger.info(f"[{self.name}] Weekly cadence trigger activated")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"[{self.name}] Error checking weekly cadence: {e}")
            return False
    
    def check_major_conflict_trigger(self) -> bool:
        """
        Check if there's a major belief conflict that should trigger evolution.
        
        Returns:
            True if major conflict evolution should be triggered
        """
        try:
            contradiction_analysis = self.analyze_contradiction_stress()
            pressure_level = contradiction_analysis.get('pressure_level', 0.0)
            
            if pressure_level >= self.major_conflict_threshold:
                logger.info(f"[{self.name}] Major conflict trigger activated (pressure: {pressure_level:.2f})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"[{self.name}] Error checking major conflict trigger: {e}")
            return False
    
    def get_personality_status(self) -> Dict[str, Any]:
        """
        Get current personality status for reporting.
        
        Returns:
            Comprehensive personality status report
        """
        try:
            traits = self.personality_vector.get_trait_dict()
            
            status = {
                'current_traits': traits,
                'personality_vector': {
                    'base_tone': self.personality_vector.base_tone,
                    'emotional_baseline': self.personality_vector.emotional_baseline,
                    'emotional_reactivity': self.personality_vector.emotional_reactivity
                },
                'evolution_metadata': {
                    'total_evolutions': self.personality_vector.evolution_count,
                    'last_evolution': self.personality_vector.last_evolution,
                    'total_evolution_magnitude': self.personality_vector.total_evolution_magnitude,
                    'last_weekly_check': self.last_weekly_check
                },
                'configuration': {
                    'sensitivity_threshold': self.sensitivity_threshold,
                    'max_shift_rate': self.max_shift_rate,
                    'trait_bounds': self.trait_bounds,
                    'weekly_cadence_hours': self.weekly_cadence_hours
                },
                'recent_evolution_history': []
            }
            
            # Add recent evolution history (last 5)
            if self.evolution_history:
                recent_traces = self.evolution_history[-5:]
                for trace_data in recent_traces:
                    status['recent_evolution_history'].append({
                        'evolution_id': trace_data.get('evolution_id'),
                        'trigger_type': trace_data.get('trigger_type'),
                        'timestamp': trace_data.get('timestamp'),
                        'total_changes': trace_data.get('total_changes', 0),
                        'evolution_magnitude': trace_data.get('evolution_magnitude', 0.0),
                        'summary': trace_data.get('natural_language_summary', 'No summary available')
                    })
            
            return status
            
        except Exception as e:
            logger.error(f"[{self.name}] Error getting personality status: {e}")
            return {'error': str(e)}
    
    def get_evolution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get personality evolution history.
        
        Args:
            limit: Maximum number of evolution traces to return
            
        Returns:
            List of evolution traces
        """
        try:
            if not self.evolution_history:
                return []
            
            # Return the most recent evolution traces
            recent_traces = self.evolution_history[-limit:] if len(self.evolution_history) > limit else self.evolution_history
            
            # Reverse to show most recent first
            return list(reversed(recent_traces))
            
        except Exception as e:
            logger.error(f"[{self.name}] Error getting evolution history: {e}")
            return []


# Singleton instance
_personality_evolver_instance = None

def get_personality_evolver() -> PersonalityEvolver:
    """Get the singleton PersonalityEvolver instance."""
    global _personality_evolver_instance
    if _personality_evolver_instance is None:
        _personality_evolver_instance = PersonalityEvolver()
    return _personality_evolver_instance