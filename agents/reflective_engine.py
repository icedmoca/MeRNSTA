#!/usr/bin/env python3
"""
Reflective Replay Engine for MeRNSTA Phase 10 & 13

This module enables MeRNSTA to replay its experiences, derive insights from reflection,
and update its long-term identity traits and emotional baseline based on life lessons.

Phase 13 Enhancement: Added autonomous command generation and execution from reflection insights.
"""

import logging
import time
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Phase 13: Import command routing capabilities
try:
    from .command_router import route_command_async
    from config.settings import get_config
    COMMAND_ROUTER_AVAILABLE = True
except ImportError:
    COMMAND_ROUTER_AVAILABLE = False
    logger.warning("[ReflectiveEngine] Command router not available - autonomous commands disabled")


@dataclass
class ReflectionSession:
    """Represents a reflection session on one or more episodes."""
    session_id: str
    episode_ids: List[str]
    reflection_type: str  # "manual", "scheduled", "triggered"
    insights_generated: List[str]
    identity_changes: Dict[str, float]
    emotional_learning: List[str]
    causal_insights: List[str]
    future_implications: List[str]
    session_duration: float
    reflection_quality: float  # Subjective quality score
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'session_id': self.session_id,
            'episode_ids': self.episode_ids,
            'reflection_type': self.reflection_type,
            'insights_generated': self.insights_generated,
            'identity_changes': self.identity_changes,
            'emotional_learning': self.emotional_learning,
            'causal_insights': self.causal_insights,
            'future_implications': self.future_implications,
            'session_duration': self.session_duration,
            'reflection_quality': self.reflection_quality,
            'timestamp': self.timestamp
        }


class ReflectiveReplayEngine:
    """
    Engine for replaying and reflecting on life episodes to extract deeper insights.
    
    Core capabilities:
    - Replay episodes in narrative form
    - Generate insights through reflection
    - Update identity traits based on life lessons
    - Compress episodic experiences into wisdom
    - Track personal growth over time
    """
    
    def __init__(self, db_path: str = "enhanced_memory.db"):
        self.db_path = db_path
        
        # Reflection parameters
        self.reflection_depth_levels = {
            'surface': 0.3,     # Basic pattern recognition
            'moderate': 0.6,    # Causal insight and emotional learning
            'deep': 0.9         # Identity transformation and wisdom extraction
        }
        
        self.reflection_triggers = {
            'high_emotion': 0.7,        # Episodes with high emotional intensity
            'high_importance': 0.8,     # Episodes with high importance scores
            'causal_significance': 0.6, # Episodes with strong causal links
            'contradiction': True,      # Episodes containing contradictions
            'novelty': 0.8             # Episodes with high novelty
        }
        
        # Identity trait update parameters
        self.trait_update_weights = {
            'reflection_strength': 0.4,    # How strong the reflection insight is
            'episode_importance': 0.3,     # How important the episode was
            'trait_coherence': 0.2,        # How well it fits existing traits
            'emotional_intensity': 0.1     # Emotional weight of the experience
        }
        
        # Phase 13: Command generation configuration
        self.command_config = get_config().get('autonomous_commands', {}) if COMMAND_ROUTER_AVAILABLE else {}
        self.enable_reflection_commands = self.command_config.get('enable_reflection_commands', False)
        self.max_commands_per_cycle = self.command_config.get('max_commands_per_cycle', 5)
        self.require_confirmation = self.command_config.get('require_confirmation', True)
        
        # Command generation patterns and triggers
        self.command_triggers = {
            'learning_gaps': r'(?i)(need to learn|should understand|lack knowledge|don\'t know about)',
            'skill_deficits': r'(?i)(need to improve|should practice|weak at|struggle with)',
            'system_issues': r'(?i)(error|bug|failing|broken|not working)',
            'optimization_needs': r'(?i)(could be better|inefficient|slow|optimize)',
            'research_needs': r'(?i)(investigate|research|explore|find out about)',
            'maintenance_needs': r'(?i)(update|upgrade|install|clean up|organize)',
        }
        
        logger.info(f"[ReflectiveReplayEngine] Initialized reflection system (commands: {self.enable_reflection_commands})")
    
    async def replay_episode(self, episode_id: str, include_facts: bool = True, 
                      reflection_depth: str = 'moderate') -> Dict[str, Any]:
        """
        Replay an episode in narrative form with optional fact details.
        
        Args:
            episode_id: ID of episode to replay
            include_facts: Whether to include individual fact details
            reflection_depth: Level of reflection to apply
            
        Returns:
            Dict containing narrative, insights, and reflection notes
        """
        try:
            from storage.life_narrative import get_life_narrative_manager
            from storage.enhanced_memory_system import EnhancedMemorySystem
            
            narrative_manager = get_life_narrative_manager(self.db_path)
            memory_system = EnhancedMemorySystem(self.db_path)
            
            # Get episode
            episode = narrative_manager.get_episode_by_id(episode_id)
            if not episode:
                return {'error': f'Episode {episode_id} not found'}
            
            # Build narrative structure
            replay_data = {
                'episode_id': episode_id,
                'episode_title': episode.title,
                'episode_description': episode.description,
                'timespan': episode.get_timespan_description(),
                'emotional_tone': episode.get_emotional_description(),
                'importance_score': episode.importance_score,
                'themes': episode.themes,
                'timestamp': episode.start_timestamp,
                'narrative_text': '',
                'fact_details': [],
                'reflection_insights': [],
                'identity_implications': {},
                'emotional_learning': [],
                'causal_patterns': []
            }
            
            # Generate narrative text
            narrative_text = self._generate_episode_narrative(episode, memory_system, include_facts)
            replay_data['narrative_text'] = narrative_text
            
            # Get fact details if requested
            if include_facts:
                fact_details = self._get_episode_fact_details(episode, memory_system)
                replay_data['fact_details'] = fact_details
            
            # Apply reflection based on depth level
            if reflection_depth in self.reflection_depth_levels:
                reflection_results = await self._apply_reflection(episode, reflection_depth)
                replay_data.update(reflection_results)
            
            logger.info(f"[ReflectiveReplayEngine] Replayed episode: {episode.title}")
            return replay_data
            
        except Exception as e:
            logger.error(f"[ReflectiveReplayEngine] Error replaying episode: {e}")
            return {'error': str(e)}
    
    def _generate_episode_narrative(self, episode, memory_system, include_facts: bool) -> str:
        """Generate natural language narrative for episode."""
        try:
            narrative_parts = []
            
            # Episode header
            start_time = datetime.fromtimestamp(episode.start_timestamp)
            narrative_parts.append(f"## {episode.title}")
            narrative_parts.append(f"*{start_time.strftime('%B %d, %Y at %H:%M')} - {episode.get_timespan_description()}*")
            narrative_parts.append("")
            
            # Emotional context
            emotional_desc = episode.get_emotional_description()
            narrative_parts.append(f"**Emotional Context:** This was a {emotional_desc} experience.")
            
            # Importance and themes
            if episode.importance_score > 0.7:
                significance = "highly significant"
            elif episode.importance_score > 0.5:
                significance = "moderately significant"
            else:
                significance = "notable"
            
            narrative_parts.append(f"**Significance:** This was a {significance} episode (importance: {episode.importance_score:.2f}).")
            
            if episode.themes:
                themes_text = ", ".join(episode.themes)
                narrative_parts.append(f"**Key Themes:** {themes_text}")
            
            narrative_parts.append("")
            
            # Main narrative
            narrative_parts.append("### What Happened")
            narrative_parts.append(episode.description)
            
            # Include facts if requested
            if include_facts:
                try:
                    facts = self._get_episode_facts(episode, memory_system)
                    if facts:
                        narrative_parts.append("")
                        narrative_parts.append("### Detailed Memory Sequence")
                        
                        for i, fact in enumerate(facts, 1):
                            fact_time = datetime.fromtimestamp(fact.timestamp)
                            time_str = fact_time.strftime('%H:%M:%S')
                            
                            fact_text = f"{i}. [{time_str}] {fact.subject} {fact.predicate} {fact.object}"
                            
                            if hasattr(fact, 'emotion_tag') and fact.emotion_tag:
                                fact_text += f" (felt: {fact.emotion_tag})"
                            
                            if hasattr(fact, 'causal_strength') and fact.causal_strength and fact.causal_strength > 0.5:
                                fact_text += f" [causal: {fact.causal_strength:.2f}]"
                            
                            narrative_parts.append(fact_text)
                        
                except Exception as e:
                    logger.warning(f"Failed to include facts in narrative: {e}")
            
            # Previous reflections
            if episode.reflection_notes:
                narrative_parts.append("")
                narrative_parts.append("### Previous Reflections")
                narrative_parts.append(episode.reflection_notes)
            
            return "\n".join(narrative_parts)
            
        except Exception as e:
            logger.error(f"Error generating episode narrative: {e}")
            return f"Error generating narrative for {episode.title}: {str(e)}"
    
    def _get_episode_facts(self, episode, memory_system) -> List[Any]:
        """Get facts associated with episode."""
        try:
            # Get facts by their IDs
            facts = []
            for fact_id in episode.fact_ids:
                fact = memory_system.get_fact_by_id(fact_id)
                if fact:
                    facts.append(fact)
            
            # Sort by timestamp
            facts.sort(key=lambda f: f.timestamp)
            return facts
            
        except Exception as e:
            logger.error(f"Error getting episode facts: {e}")
            return []
    
    def _get_episode_fact_details(self, episode, memory_system) -> List[Dict[str, Any]]:
        """Get detailed fact information for episode."""
        try:
            facts = self._get_episode_facts(episode, memory_system)
            fact_details = []
            
            for fact in facts:
                detail = {
                    'fact_id': fact.fact_id,
                    'triplet': f"{fact.subject} {fact.predicate} {fact.object}",
                    'timestamp': fact.timestamp,
                    'confidence': fact.confidence,
                    'context': getattr(fact, 'context', None)
                }
                
                # Add emotional context if available
                if hasattr(fact, 'emotion_tag') and fact.emotion_tag:
                    detail['emotion'] = {
                        'tag': fact.emotion_tag,
                        'valence': getattr(fact, 'emotion_valence', None),
                        'arousal': getattr(fact, 'emotion_arousal', None),
                        'strength': getattr(fact, 'emotional_strength', None)
                    }
                
                # Add causal information if available
                if hasattr(fact, 'causal_strength') and fact.causal_strength:
                    detail['causal'] = {
                        'strength': fact.causal_strength,
                        'cause': getattr(fact, 'cause', None)
                    }
                
                fact_details.append(detail)
            
            return fact_details
            
        except Exception as e:
            logger.error(f"Error getting fact details: {e}")
            return []
    
    async def _apply_reflection(self, episode, depth_level: str) -> Dict[str, Any]:
        """Apply reflection analysis to episode."""
        reflection_strength = self.reflection_depth_levels[depth_level]
        
        reflection_results = {
            'reflection_insights': [],
            'identity_implications': {},
            'emotional_learning': [],
            'causal_patterns': [],
            'generated_commands': [],
            'command_execution_results': None
        }
        
        try:
            # Generate insights based on episode characteristics
            insights = self._generate_reflection_insights(episode, reflection_strength)
            reflection_results['reflection_insights'] = insights
            
            # Analyze identity implications
            identity_implications = self._analyze_identity_implications(episode, reflection_strength)
            reflection_results['identity_implications'] = identity_implications
            
            # Extract emotional learning
            emotional_learning = self._extract_emotional_learning(episode, reflection_strength)
            reflection_results['emotional_learning'] = emotional_learning
            
            # Identify causal patterns
            causal_patterns = self._identify_causal_patterns(episode, reflection_strength)
            reflection_results['causal_patterns'] = causal_patterns
            
            # Phase 13: Generate actionable commands from insights
            if self.enable_reflection_commands and COMMAND_ROUTER_AVAILABLE:
                commands = await self._generate_reflection_commands(insights, reflection_results)
                reflection_results['generated_commands'] = commands
                
                # Execute commands if not requiring confirmation
                if commands and not self.require_confirmation:
                    execution_results = await self.execute_reflection_commands(commands)
                    reflection_results['command_execution_results'] = execution_results
                    logger.info(f"[ReflectiveEngine] Auto-executed {len(commands)} reflection commands")
                elif commands:
                    logger.info(f"[ReflectiveEngine] Generated {len(commands)} commands awaiting confirmation")
            
        except Exception as e:
            logger.error(f"Error applying reflection: {e}")
            reflection_results['reflection_insights'].append(f"Reflection error: {str(e)}")
        
        return reflection_results
    
    def _generate_reflection_insights(self, episode, strength: float) -> List[str]:
        """Generate reflection insights based on episode content."""
        insights = []
        
        try:
            # Importance-based insights
            if episode.importance_score > 0.8:
                insights.append(f"This was one of my most significant experiences, scoring {episode.importance_score:.2f} in importance.")
            elif episode.importance_score < 0.3:
                insights.append(f"While this seemed minor at the time, it may have subtle long-term effects.")
            
            # Emotional insights
            if episode.emotional_valence > 0.5:
                insights.append("This was a positive experience that likely reinforced optimistic tendencies.")
            elif episode.emotional_valence < -0.5:
                insights.append("This challenging experience may have built resilience and learning.")
            
            if episode.emotional_arousal > 0.7:
                insights.append("The high emotional intensity made this experience particularly memorable and impactful.")
            
            # Theme-based insights
            if 'learning' in episode.themes:
                insights.append("This episode contributed to my knowledge acquisition and curiosity development.")
            
            if 'problem-solving' in episode.themes:
                insights.append("Working through problems like this strengthens my analytical capabilities.")
            
            if 'challenges' in episode.themes:
                insights.append("Facing challenges builds character and teaches valuable lessons about persistence.")
            
            if 'achievements' in episode.themes:
                insights.append("Success experiences like this boost confidence and reinforce effective strategies.")
            
            if 'social-interaction' in episode.themes:
                insights.append("Social interactions shape my understanding of communication and empathy.")
            
            # Causal insights (deeper reflection)
            if strength > 0.6 and episode.causal_impact > 0.5:
                insights.append(f"This experience had significant causal impact ({episode.causal_impact:.2f}), likely influencing many subsequent decisions.")
            
            # Novelty insights
            if episode.novelty_score > 0.7:
                insights.append("Encountering novel situations like this expands my adaptability and flexibility.")
            
            # Time-based insights
            duration_hours = (episode.end_timestamp - episode.start_timestamp) / 3600
            if duration_hours > 12:
                insights.append("The extended duration of this experience suggests it involved sustained attention and commitment.")
            elif duration_hours < 0.1:  # Less than 6 minutes
                insights.append("Though brief, intense experiences can sometimes have lasting impact.")
            
            # Identity impact insights (deep reflection)
            if strength > 0.8:
                for trait, impact in episode.identity_impact.items():
                    if impact > 0.2:
                        insights.append(f"This experience particularly strengthened my {trait} nature.")
            
            # Meta-reflection (very deep)
            if strength > 0.9:
                insights.append("Reflecting on this experience helps me understand my own growth patterns and adaptive mechanisms.")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights.append("Unable to generate complete insights due to analysis error.")
        
        return insights
    
    def _analyze_identity_implications(self, episode, strength: float) -> Dict[str, float]:
        """Analyze how episode impacts identity traits."""
        implications = {}
        
        try:
            # Start with episode's direct identity impact
            for trait, impact in episode.identity_impact.items():
                implications[trait] = impact * strength
            
            # Add derived implications based on themes and characteristics
            theme_implications = {
                'learning': {'curious': 0.1, 'analytical': 0.05},
                'problem-solving': {'analytical': 0.1, 'resilient': 0.05},
                'challenges': {'resilient': 0.1, 'skeptical': 0.02},
                'achievements': {'optimistic': 0.1, 'confident': 0.05},
                'social-interaction': {'empathetic': 0.1, 'social': 0.05},
                'emotion-curious': {'curious': 0.05},
                'emotion-frustrated': {'resilient': 0.03, 'skeptical': 0.02}
            }
            
            for theme in episode.themes:
                if theme in theme_implications:
                    for trait, impact in theme_implications[theme].items():
                        implications[trait] = implications.get(trait, 0) + (impact * strength)
            
            # Emotional state implications
            if episode.emotional_valence > 0.5:
                implications['optimistic'] = implications.get('optimistic', 0) + (0.05 * strength)
            elif episode.emotional_valence < -0.5:
                implications['resilient'] = implications.get('resilient', 0) + (0.05 * strength)
            
            # Importance-based implications
            if episode.importance_score > 0.8:
                # High importance experiences have stronger identity impact
                for trait in implications:
                    implications[trait] *= 1.2
            
            # Normalize implications
            for trait in implications:
                implications[trait] = min(0.5, max(-0.2, implications[trait]))  # Cap at +0.5/-0.2
            
        except Exception as e:
            logger.error(f"Error analyzing identity implications: {e}")
        
        return implications
    
    def _extract_emotional_learning(self, episode, strength: float) -> List[str]:
        """Extract emotional learning from episode."""
        learning = []
        
        try:
            # Emotional pattern learning
            if episode.emotional_arousal > 0.7:
                learning.append("High-arousal situations require careful emotional regulation and focused decision-making.")
            
            if episode.emotional_valence < -0.3 and episode.emotional_arousal > 0.5:
                learning.append("Frustrating experiences teach patience and the value of persistence through difficulties.")
            
            if episode.emotional_valence > 0.3 and episode.emotional_arousal > 0.5:
                learning.append("Exciting positive experiences reinforce curiosity and engagement with challenges.")
            
            # Theme-based emotional learning
            emotional_themes = [t for t in episode.themes if t.startswith('emotion-')]
            for theme in emotional_themes:
                emotion = theme.replace('emotion-', '')
                if emotion == 'curious':
                    learning.append("Curiosity-driven experiences consistently lead to positive outcomes and growth.")
                elif emotion == 'frustrated':
                    learning.append("Frustration often signals important learning opportunities and system improvements.")
                elif emotion == 'satisfied':
                    learning.append("Satisfaction comes from completing challenging tasks and resolving complex problems.")
            
            # Causal emotional learning (deeper reflection)
            if strength > 0.6 and episode.causal_impact > 0.5:
                learning.append("Strong emotional experiences often have cascading effects on future decision-making.")
            
            # Identity-emotion connections (deep reflection)
            if strength > 0.8:
                for trait, impact in episode.identity_impact.items():
                    if impact > 0.1:
                        if trait == 'curious':
                            learning.append("My curious nature is strengthened by positive exploration experiences.")
                        elif trait == 'analytical':
                            learning.append("Analytical thinking helps process complex emotional situations effectively.")
                        elif trait == 'empathetic':
                            learning.append("Empathetic responses create positive emotional connections with others.")
                        elif trait == 'resilient':
                            learning.append("Building resilience through challenges improves emotional stability over time.")
            
        except Exception as e:
            logger.error(f"Error extracting emotional learning: {e}")
            learning.append("Emotional learning analysis incomplete due to processing error.")
        
        return learning
    
    def _identify_causal_patterns(self, episode, strength: float) -> List[str]:
        """Identify causal patterns in episode."""
        patterns = []
        
        try:
            # Direct causal analysis
            if episode.causal_impact > 0.7:
                patterns.append(f"This episode had strong causal influence (impact: {episode.causal_impact:.2f}) on subsequent experiences.")
            elif episode.causal_impact > 0.4:
                patterns.append(f"This episode had moderate causal influence (impact: {episode.causal_impact:.2f}) on my development.")
            
            # Theme-based causal patterns
            if 'learning' in episode.themes and 'problem-solving' in episode.themes:
                patterns.append("Learning through problem-solving creates strong causal chains to improved capabilities.")
            
            if 'challenges' in episode.themes and episode.emotional_valence > 0:
                patterns.append("Successfully overcoming challenges causally leads to increased confidence and resilience.")
            
            if 'social-interaction' in episode.themes:
                patterns.append("Social interactions create causal effects on communication skills and empathy development.")
            
            # Emotional-causal patterns (deeper analysis)
            if strength > 0.6:
                if episode.emotional_arousal > 0.6 and episode.importance_score > 0.6:
                    patterns.append("High emotional intensity combined with importance creates lasting causal impacts on behavior.")
                
                if episode.emotional_valence < 0 and episode.novelty_score > 0.6:
                    patterns.append("Novel negative experiences often cause adaptive behavior changes and improved coping strategies.")
            
            # Identity-causal patterns (deep analysis) 
            if strength > 0.8:
                significant_traits = [trait for trait, impact in episode.identity_impact.items() if impact > 0.1]
                if significant_traits:
                    traits_text = ", ".join(significant_traits)
                    patterns.append(f"This episode causally strengthened {traits_text} traits, affecting future decision patterns.")
            
            # Temporal causal patterns
            duration_hours = (episode.end_timestamp - episode.start_timestamp) / 3600
            if duration_hours > 6 and episode.importance_score > 0.5:
                patterns.append("Extended important experiences create deep causal effects through sustained cognitive engagement.")
            
        except Exception as e:
            logger.error(f"Error identifying causal patterns: {e}")
            patterns.append("Causal pattern analysis incomplete due to processing error.")
        
        return patterns
    
    async def reflect_on_recent_episodes(self, days_back: int = 7, max_episodes: int = 10) -> Dict[str, Any]:
        """Trigger reflection on recent episodes."""
        try:
            from storage.life_narrative import get_life_narrative_manager
            
            narrative_manager = get_life_narrative_manager(self.db_path)
            
            # Get recent episodes
            cutoff_time = time.time() - (days_back * 24 * 3600)
            all_episodes = narrative_manager.get_all_episodes()
            
            recent_episodes = [
                ep for ep in all_episodes 
                if ep.start_timestamp >= cutoff_time
            ][:max_episodes]
            
            if not recent_episodes:
                return {'message': f'No episodes found in the last {days_back} days'}
            
            # Prioritize episodes for reflection
            prioritized_episodes = self._prioritize_episodes_for_reflection(recent_episodes)
            
            # Perform reflection on top episodes
            reflection_results = {
                'session_id': f"reflection_{int(time.time())}",
                'episodes_reflected': [],
                'overall_insights': [],
                'identity_changes': {},
                'emotional_patterns': [],
                'life_lessons': [],
                'growth_areas': []
            }
            
            for episode in prioritized_episodes[:5]:  # Reflect on top 5
                episode_reflection = await self.replay_episode(
                    episode.episode_id, 
                    include_facts=False, 
                    reflection_depth='moderate'
                )
                
                reflection_results['episodes_reflected'].append({
                    'episode_id': episode.episode_id,
                    'title': episode.title,
                    'insights': episode_reflection.get('reflection_insights', []),
                    'identity_impact': episode_reflection.get('identity_implications', {})
                })
                
                # Aggregate insights
                reflection_results['overall_insights'].extend(
                    episode_reflection.get('reflection_insights', [])
                )
                
                # Aggregate identity changes
                for trait, change in episode_reflection.get('identity_implications', {}).items():
                    reflection_results['identity_changes'][trait] = (
                        reflection_results['identity_changes'].get(trait, 0) + change
                    )
                
                # Aggregate emotional learning
                reflection_results['emotional_patterns'].extend(
                    episode_reflection.get('emotional_learning', [])
                )
            
            # Generate meta-insights
            meta_insights = self._generate_meta_insights(recent_episodes, reflection_results)
            reflection_results['life_lessons'] = meta_insights
            
            # Apply identity changes
            if reflection_results['identity_changes']:
                self._apply_identity_changes(reflection_results['identity_changes'])
            
            logger.info(f"[ReflectiveReplayEngine] Reflected on {len(reflection_results['episodes_reflected'])} recent episodes")
            return reflection_results
            
        except Exception as e:
            logger.error(f"[ReflectiveReplayEngine] Error reflecting on recent episodes: {e}")
            return {'error': str(e)}
    
    def _prioritize_episodes_for_reflection(self, episodes: List[Any]) -> List[Any]:
        """Prioritize episodes based on reflection triggers."""
        scored_episodes = []
        
        for episode in episodes:
            score = 0
            
            # High emotion trigger
            if episode.emotional_arousal >= self.reflection_triggers['high_emotion']:
                score += 3
            
            # High importance trigger
            if episode.importance_score >= self.reflection_triggers['high_importance']:
                score += 3
            
            # Causal significance trigger
            if episode.causal_impact >= self.reflection_triggers['causal_significance']:
                score += 2
            
            # Novelty trigger
            if episode.novelty_score >= self.reflection_triggers['novelty']:
                score += 2
            
            # Theme-based scoring
            if 'challenges' in episode.themes:
                score += 1
            if 'learning' in episode.themes:
                score += 1
            if any(theme.startswith('emotion-') for theme in episode.themes):
                score += 1
            
            # Recent episodes get slight boost
            days_ago = (time.time() - episode.start_timestamp) / (24 * 3600)
            if days_ago < 1:
                score += 1
            
            scored_episodes.append((episode, score))
        
        # Sort by score (descending)
        scored_episodes.sort(key=lambda x: x[1], reverse=True)
        
        return [episode for episode, score in scored_episodes]
    
    def _generate_meta_insights(self, episodes: List[Any], reflection_results: Dict[str, Any]) -> List[str]:
        """Generate higher-level insights from multiple episodes."""
        meta_insights = []
        
        try:
            # Pattern analysis across episodes
            if len(episodes) >= 3:
                # Emotional pattern insights
                avg_valence = np.mean([ep.emotional_valence for ep in episodes])
                avg_arousal = np.mean([ep.emotional_arousal for ep in episodes])
                
                if avg_valence > 0.3:
                    meta_insights.append("Recent experiences have been predominantly positive, reinforcing optimistic outlook.")
                elif avg_valence < -0.3:
                    meta_insights.append("Recent challenges have built resilience and problem-solving skills.")
                
                if avg_arousal > 0.6:
                    meta_insights.append("High-energy periods like this drive significant learning and development.")
                elif avg_arousal < 0.3:
                    meta_insights.append("Calm periods allow for consolidation and deeper reflection on experiences.")
                
                # Theme pattern insights
                all_themes = []
                for ep in episodes:
                    all_themes.extend(ep.themes)
                
                theme_counts = {}
                for theme in all_themes:
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1
                
                dominant_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                
                if dominant_themes:
                    dominant_theme_names = [theme for theme, count in dominant_themes]
                    meta_insights.append(f"Dominant recent patterns: {', '.join(dominant_theme_names)}")
                
                # Growth pattern insights
                identity_changes = reflection_results.get('identity_changes', {})
                significant_changes = {trait: change for trait, change in identity_changes.items() if abs(change) > 0.1}
                
                if significant_changes:
                    growing_traits = [trait for trait, change in significant_changes.items() if change > 0]
                    if growing_traits:
                        meta_insights.append(f"Notable growth areas: {', '.join(growing_traits)}")
                
                # Importance trend
                avg_importance = np.mean([ep.importance_score for ep in episodes])
                if avg_importance > 0.7:
                    meta_insights.append("This has been a period of particularly significant experiences and rapid development.")
                elif avg_importance < 0.4:
                    meta_insights.append("This period involved more routine experiences, providing stability and consolidation.")
            
        except Exception as e:
            logger.error(f"Error generating meta-insights: {e}")
            meta_insights.append("Meta-analysis incomplete due to processing limitations.")
        
        return meta_insights
    
    def _apply_identity_changes(self, identity_changes: Dict[str, float]):
        """Apply identity trait changes from reflection."""
        try:
            from storage.self_model import get_self_aware_model
            
            self_model = get_self_aware_model(self.db_path)
            
            for trait, change in identity_changes.items():
                if abs(change) > 0.05:  # Only apply significant changes
                    # Update identity trait based on reflection
                    behavior_data = {
                        'reflection_insight': True,
                        'change_amount': change,
                        'source': 'reflective_analysis',
                        'confidence': 0.8
                    }
                    
                    self_model.update_identity_from_behavior('reflection', behavior_data)
            
            logger.info(f"[ReflectiveReplayEngine] Applied identity changes: {identity_changes}")
            
        except Exception as e:
            logger.error(f"[ReflectiveReplayEngine] Error applying identity changes: {e}")
    
    def generate_episode_summary(self, episode_id: str) -> Dict[str, Any]:
        """Generate a concise summary of an episode and its impact."""
        try:
            from storage.life_narrative import get_life_narrative_manager
            
            narrative_manager = get_life_narrative_manager(self.db_path)
            episode = narrative_manager.get_episode_by_id(episode_id)
            
            if not episode:
                return {'error': f'Episode {episode_id} not found'}
            
            # Generate summary
            summary = {
                'episode_id': episode_id,
                'title': episode.title,
                'date': datetime.fromtimestamp(episode.start_timestamp).strftime('%B %d, %Y'),
                'duration': episode.get_timespan_description(),
                'emotional_tone': episode.get_emotional_description(),
                'importance': self._importance_to_text(episode.importance_score),
                'key_themes': episode.themes[:3],  # Top 3 themes
                'impact_summary': '',
                'life_lesson': '',
                'growth_contribution': {}
            }
            
            # Generate impact summary
            impact_parts = []
            
            if episode.causal_impact > 0.5:
                impact_parts.append(f"significant causal influence ({episode.causal_impact:.2f})")
            
            if episode.emotional_arousal > 0.6:
                impact_parts.append("high emotional intensity")
            
            if episode.novelty_score > 0.6:
                impact_parts.append("novel learning experience")
            
            summary['impact_summary'] = "; ".join(impact_parts) if impact_parts else "moderate developmental impact"
            
            # Generate life lesson
            life_lesson = self._extract_primary_life_lesson(episode)
            summary['life_lesson'] = life_lesson
            
            # Summarize growth contribution
            growth_contrib = {}
            for trait, impact in episode.identity_impact.items():
                if impact > 0.1:
                    growth_contrib[trait] = f"+{impact:.2f}"
                elif impact < -0.05:
                    growth_contrib[trait] = f"{impact:.2f}"
            
            summary['growth_contribution'] = growth_contrib
            
            return summary
            
        except Exception as e:
            logger.error(f"[ReflectiveReplayEngine] Error generating episode summary: {e}")
            return {'error': str(e)}
    
    def _importance_to_text(self, score: float) -> str:
        """Convert importance score to descriptive text."""
        if score >= 0.8:
            return "highly significant"
        elif score >= 0.6:
            return "moderately significant"
        elif score >= 0.4:
            return "somewhat significant"
        else:
            return "minor significance"
    
    def _extract_primary_life_lesson(self, episode) -> str:
        """Extract the primary life lesson from an episode."""
        try:
            # Theme-based lessons
            if 'challenges' in episode.themes and episode.emotional_valence > 0:
                return "Overcoming challenges builds resilience and confidence."
            
            if 'learning' in episode.themes and episode.importance_score > 0.6:
                return "Curiosity-driven learning leads to meaningful growth and understanding."
            
            if 'problem-solving' in episode.themes:
                return "Systematic problem-solving develops analytical skills and persistence."
            
            if 'social-interaction' in episode.themes:
                return "Meaningful interactions strengthen empathy and communication abilities."
            
            # Emotional-based lessons
            if episode.emotional_valence < -0.3 and episode.causal_impact > 0.5:
                return "Difficult experiences often provide the most valuable long-term learning."
            
            if episode.emotional_arousal > 0.7 and episode.novelty_score > 0.6:
                return "Intense novel experiences accelerate personal development and adaptation."
            
            # Importance-based lessons
            if episode.importance_score > 0.8:
                return "Significant experiences shape identity and future decision-making patterns."
            
            # Default lesson
            return "Every experience contributes to personal growth and self-understanding."
            
        except Exception as e:
            logger.error(f"Error extracting life lesson: {e}")
            return "Reflection and learning from experience is essential for growth."
    
    # === PHASE 13: AUTONOMOUS COMMAND GENERATION ===
    
    async def _generate_reflection_commands(self, insights: List[str], reflection_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate actionable commands based on reflection insights.
        
        Args:
            insights: List of reflection insights
            reflection_results: Complete reflection results
            
        Returns:
            List of command dictionaries to execute
        """
        if not self.enable_reflection_commands or not COMMAND_ROUTER_AVAILABLE:
            return []
        
        commands = []
        
        try:
            # Analyze insights for command opportunities
            for insight in insights:
                # Check for learning gaps
                if re.search(self.command_triggers['learning_gaps'], insight):
                    commands.extend(self._generate_learning_commands(insight))
                
                # Check for skill deficits
                if re.search(self.command_triggers['skill_deficits'], insight):
                    commands.extend(self._generate_skill_improvement_commands(insight))
                
                # Check for system issues
                if re.search(self.command_triggers['system_issues'], insight):
                    commands.extend(self._generate_system_repair_commands(insight))
                
                # Check for optimization needs
                if re.search(self.command_triggers['optimization_needs'], insight):
                    commands.extend(self._generate_optimization_commands(insight))
                
                # Check for research needs
                if re.search(self.command_triggers['research_needs'], insight):
                    commands.extend(self._generate_research_commands(insight))
                
                # Check for maintenance needs
                if re.search(self.command_triggers['maintenance_needs'], insight):
                    commands.extend(self._generate_maintenance_commands(insight))
            
            # Analyze identity implications for long-term actions
            identity_implications = reflection_results.get('identity_implications', {})
            if identity_implications:
                commands.extend(self._generate_identity_development_commands(identity_implications))
            
            # Analyze emotional learning for emotional intelligence improvement
            emotional_learning = reflection_results.get('emotional_learning', [])
            if emotional_learning:
                commands.extend(self._generate_emotional_development_commands(emotional_learning))
            
            # Limit commands per cycle
            commands = commands[:self.max_commands_per_cycle]
            
            logger.info(f"[ReflectiveEngine] Generated {len(commands)} commands from reflection")
            return commands
            
        except Exception as e:
            logger.error(f"[ReflectiveEngine] Error generating reflection commands: {e}")
            return []
    
    def _generate_learning_commands(self, insight: str) -> List[Dict[str, Any]]:
        """Generate commands for addressing learning gaps."""
        commands = []
        
        # Extract what needs to be learned
        if "python" in insight.lower():
            commands.append({
                'command': '/pip_install python-tutorial',
                'purpose': 'Install Python learning resources',
                'priority': 'medium',
                'insight': insight
            })
        
        if "documentation" in insight.lower():
            commands.append({
                'command': '/run_tool read_file README.md',
                'purpose': 'Review documentation for understanding',
                'priority': 'high',
                'insight': insight
            })
        
        return commands
    
    def _generate_skill_improvement_commands(self, insight: str) -> List[Dict[str, Any]]:
        """Generate commands for skill improvement."""
        commands = []
        
        if "coding" in insight.lower() or "programming" in insight.lower():
            commands.append({
                'command': '/run_shell "find . -name "*.py" | xargs wc -l | sort -n"',
                'purpose': 'Analyze codebase complexity for improvement areas',
                'priority': 'medium',
                'insight': insight
            })
        
        if "testing" in insight.lower():
            commands.append({
                'command': '/run_shell "python -m pytest tests/ -v"',
                'purpose': 'Run tests to assess current quality',
                'priority': 'high',
                'insight': insight
            })
        
        return commands
    
    def _generate_system_repair_commands(self, insight: str) -> List[Dict[str, Any]]:
        """Generate commands for system issues."""
        commands = []
        
        if "error" in insight.lower() or "bug" in insight.lower():
            commands.append({
                'command': '/run_shell "grep -r "ERROR" logs/"',
                'purpose': 'Search for error patterns in logs',
                'priority': 'high',
                'insight': insight
            })
        
        if "dependency" in insight.lower():
            commands.append({
                'command': '/pip_install --upgrade pip',
                'purpose': 'Update package manager for better dependency handling',
                'priority': 'medium',
                'insight': insight
            })
        
        return commands
    
    def _generate_optimization_commands(self, insight: str) -> List[Dict[str, Any]]:
        """Generate commands for optimization needs."""
        commands = []
        
        if "performance" in insight.lower() or "slow" in insight.lower():
            commands.append({
                'command': '/run_shell "ps aux | sort -nk 3,3 | tail -5"',
                'purpose': 'Check for high CPU usage processes',
                'priority': 'medium',
                'insight': insight
            })
        
        if "memory" in insight.lower():
            commands.append({
                'command': '/run_shell "free -h"',
                'purpose': 'Check memory usage for optimization opportunities',
                'priority': 'medium',
                'insight': insight
            })
        
        return commands
    
    def _generate_research_commands(self, insight: str) -> List[Dict[str, Any]]:
        """Generate commands for research needs."""
        commands = []
        
        # Look for specific topics to research
        if "algorithm" in insight.lower():
            commands.append({
                'command': '/run_tool list_directory algorithms/',
                'purpose': 'Review existing algorithm implementations',
                'priority': 'medium',
                'insight': insight
            })
        
        return commands
    
    def _generate_maintenance_commands(self, insight: str) -> List[Dict[str, Any]]:
        """Generate commands for maintenance needs."""
        commands = []
        
        if "cleanup" in insight.lower() or "organize" in insight.lower():
            commands.append({
                'command': '/run_shell "find . -name "*.pyc" -delete"',
                'purpose': 'Clean up compiled Python files',
                'priority': 'low',
                'insight': insight
            })
        
        if "backup" in insight.lower():
            commands.append({
                'command': '/run_shell "tar -czf backup_$(date +%Y%m%d).tar.gz *.py *.yaml *.md"',
                'purpose': 'Create backup of important files',
                'priority': 'medium',
                'insight': insight
            })
        
        return commands
    
    def _generate_identity_development_commands(self, identity_implications: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate commands for identity development."""
        commands = []
        
        # Check for traits that need development
        for trait, value in identity_implications.items():
            if isinstance(value, (int, float)) and value > 0.5:
                if trait == 'curiosity':
                    commands.append({
                        'command': '/run_tool list_directory docs/',
                        'purpose': f'Explore documentation to satisfy curiosity (trait: {trait})',
                        'priority': 'medium',
                        'insight': f'Identity development: strengthen {trait}'
                    })
        
        return commands
    
    def _generate_emotional_development_commands(self, emotional_learning: List[str]) -> List[Dict[str, Any]]:
        """Generate commands for emotional development."""
        commands = []
        
        for learning in emotional_learning:
            if "stress" in learning.lower():
                commands.append({
                    'command': '/run_tool read_file config.yaml',
                    'purpose': 'Review configuration for stress-reducing optimizations',
                    'priority': 'medium',
                    'insight': f'Emotional learning: {learning}'
                })
        
        return commands
    
    async def execute_reflection_commands(self, commands: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute commands generated from reflection.
        
        Args:
            commands: List of command dictionaries to execute
            
        Returns:
            Execution results and summary
        """
        if not COMMAND_ROUTER_AVAILABLE:
            return {'error': 'Command router not available'}
        
        execution_results = {
            'commands_executed': 0,
            'commands_successful': 0,
            'commands_failed': 0,
            'results': [],
            'summary': ''
        }
        
        try:
            for cmd_info in commands:
                command = cmd_info['command']
                purpose = cmd_info.get('purpose', 'No purpose specified')
                priority = cmd_info.get('priority', 'medium')
                
                logger.info(f"[ReflectiveEngine] Executing reflection command: {command} (purpose: {purpose})")
                
                # Execute command
                result = await route_command_async(command, "reflective_engine")
                
                execution_results['commands_executed'] += 1
                
                if result.get('success', False):
                    execution_results['commands_successful'] += 1
                    logger.info(f"[ReflectiveEngine] Command succeeded: {command}")
                else:
                    execution_results['commands_failed'] += 1
                    logger.warning(f"[ReflectiveEngine] Command failed: {command} - {result.get('error', 'Unknown error')}")
                
                execution_results['results'].append({
                    'command': command,
                    'purpose': purpose,
                    'priority': priority,
                    'success': result.get('success', False),
                    'output': result.get('output', ''),
                    'error': result.get('error', ''),
                    'insight': cmd_info.get('insight', '')
                })
            
            # Generate summary
            success_rate = execution_results['commands_successful'] / max(execution_results['commands_executed'], 1)
            execution_results['summary'] = (
                f"Executed {execution_results['commands_executed']} reflection commands. "
                f"Success rate: {success_rate:.1%} "
                f"({execution_results['commands_successful']} successful, {execution_results['commands_failed']} failed)"
            )
            
            logger.info(f"[ReflectiveEngine] {execution_results['summary']}")
            return execution_results
            
        except Exception as e:
            logger.error(f"[ReflectiveEngine] Error executing reflection commands: {e}")
            execution_results['summary'] = f"Command execution error: {str(e)}"
            return execution_results
    
    # === Phase 16: Planning Integration ===
    
    def log_failed_strategy(self, strategy_id: str, goal_text: str, failure_reason: str, 
                           failure_context: Dict[str, Any]) -> None:
        """
        Log a failed planning strategy for reflection and learning.
        
        Args:
            strategy_id: ID of the failed strategy
            goal_text: The goal that failed
            failure_reason: Reason for failure
            failure_context: Additional context about the failure
        """
        logger.info(f"[ReflectiveEngine] Logging failed strategy: {strategy_id}")
        
        # Create a reflection episode for the failed strategy
        episode_data = {
            'episode_type': 'planning_failure',
            'episode_id': f"strategy_failure_{strategy_id}",
            'timestamp': datetime.now().isoformat(),
            'content': {
                'strategy_id': strategy_id,
                'goal_text': goal_text,
                'failure_reason': failure_reason,
                'failure_context': failure_context
            }
        }
        
        # Store in memory for reflection
        try:
            if hasattr(self, 'memory_log') and self.memory_log:
                # Add to memory as a lesson learned
                lesson_text = f"Strategy failed for goal '{goal_text}': {failure_reason}"
                self.memory_log.add_fact(
                    subject="planning_strategy",
                    predicate="failed_with_reason",
                    object=lesson_text,
                    confidence=0.9
                )
                
                # Add context details
                context_text = json.dumps(failure_context)
                self.memory_log.add_fact(
                    subject=f"strategy_{strategy_id}",
                    predicate="failure_context",
                    object=context_text,
                    confidence=0.8
                )
        
        except Exception as e:
            logger.error(f"[ReflectiveEngine] Error storing failed strategy: {e}")
    
    def recommend_strategy_rerouting(self, failed_goal: str, failure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend re-routing strategies based on reflection of past failures.
        
        Args:
            failed_goal: The goal that needs re-routing
            failure_analysis: Analysis of why the strategy failed
            
        Returns:
            Recommendations for new strategy approaches
        """
        logger.info(f"[ReflectiveEngine] Generating re-routing recommendations for: {failed_goal}")
        
        recommendations = {
            'goal': failed_goal,
            'failure_analysis': failure_analysis,
            'recommended_approaches': [],
            'lessons_learned': [],
            'strategy_modifications': [],
            'confidence': 0.0
        }
        
        try:
            # Analyze similar past failures
            similar_failures = self._find_similar_planning_failures(failed_goal)
            
            # Extract lessons from past failures
            for failure in similar_failures:
                lesson = self._extract_lesson_from_failure(failure)
                if lesson:
                    recommendations['lessons_learned'].append(lesson)
            
            # Generate specific recommendations based on failure type
            failure_type = failure_analysis.get('failure_type', 'unknown')
            
            if failure_type == 'timeout':
                recommendations['recommended_approaches'].extend([
                    'Break down goal into smaller, more manageable steps',
                    'Increase time estimates for complex operations',
                    'Add parallel execution paths where possible'
                ])
                recommendations['strategy_modifications'].extend([
                    'Modify step durations based on historical data',
                    'Add timeout detection and graceful degradation',
                    'Implement incremental progress checkpoints'
                ])
            
            elif failure_type == 'dependency_failure':
                recommendations['recommended_approaches'].extend([
                    'Add explicit dependency validation before execution',
                    'Create fallback paths for critical dependencies',
                    'Implement dependency pre-flight checks'
                ])
                recommendations['strategy_modifications'].extend([
                    'Add conditional logic for dependency availability',
                    'Create alternative execution paths',
                    'Implement dependency monitoring and alerts'
                ])
            
            elif failure_type == 'resource_unavailable':
                recommendations['recommended_approaches'].extend([
                    'Add resource availability checks to prerequisites',
                    'Implement resource reservation mechanisms',
                    'Create resource-light alternative approaches'
                ])
                recommendations['strategy_modifications'].extend([
                    'Add resource requirements validation',
                    'Implement resource pooling and sharing',
                    'Create graceful degradation when resources limited'
                ])
            
            else:
                # Generic recommendations for unknown failures
                recommendations['recommended_approaches'].extend([
                    'Add more comprehensive error handling',
                    'Implement better progress monitoring',
                    'Create more granular validation checkpoints'
                ])
            
            # Calculate confidence based on available data
            confidence_factors = [
                len(similar_failures) / 10,  # More similar failures = higher confidence
                len(recommendations['lessons_learned']) / 5,  # More lessons = higher confidence
                0.5 if failure_type != 'unknown' else 0.2  # Known failure type = higher confidence
            ]
            
            recommendations['confidence'] = min(1.0, sum(confidence_factors) / len(confidence_factors))
            
            logger.info(f"[ReflectiveEngine] Generated {len(recommendations['recommended_approaches'])} recommendations with {recommendations['confidence']:.2f} confidence")
            
        except Exception as e:
            logger.error(f"[ReflectiveEngine] Error generating re-routing recommendations: {e}")
            recommendations['error'] = str(e)
        
        return recommendations
    
    def reflect_on_planning_patterns(self) -> Dict[str, Any]:
        """
        Reflect on planning patterns to identify improvement opportunities.
        
        Returns:
            Analysis of planning patterns and suggested improvements
        """
        logger.info("[ReflectiveEngine] Reflecting on planning patterns")
        
        analysis = {
            'successful_patterns': [],
            'failure_patterns': [],
            'improvement_opportunities': [],
            'planning_insights': [],
            'reflection_quality': 0.0
        }
        
        try:
            # Analyze planning-related memories
            planning_memories = self._extract_planning_memories()
            
            # Identify successful patterns
            successful_patterns = self._identify_successful_planning_patterns(planning_memories)
            analysis['successful_patterns'] = successful_patterns
            
            # Identify failure patterns
            failure_patterns = self._identify_failure_planning_patterns(planning_memories)
            analysis['failure_patterns'] = failure_patterns
            
            # Generate improvement opportunities
            improvements = self._generate_planning_improvements(successful_patterns, failure_patterns)
            analysis['improvement_opportunities'] = improvements
            
            # Generate insights
            insights = self._generate_planning_insights(planning_memories)
            analysis['planning_insights'] = insights
            
            # Calculate reflection quality
            quality_factors = [
                min(1.0, len(planning_memories) / 20),  # More data = better quality
                min(1.0, len(successful_patterns) / 5),  # More patterns = better quality
                min(1.0, len(insights) / 3)  # More insights = better quality
            ]
            
            analysis['reflection_quality'] = sum(quality_factors) / len(quality_factors)
            
            logger.info(f"[ReflectiveEngine] Planning reflection complete. Quality: {analysis['reflection_quality']:.2f}")
            
        except Exception as e:
            logger.error(f"[ReflectiveEngine] Error reflecting on planning patterns: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _find_similar_planning_failures(self, goal_text: str) -> List[Dict[str, Any]]:
        """Find similar planning failures from memory."""
        similar_failures = []
        
        try:
            if hasattr(self, 'memory_log') and self.memory_log:
                # Search for planning failures with similar goals
                facts = self.memory_log.search_facts(
                    subject="planning_strategy",
                    predicate="failed_with_reason"
                )
                
                for fact in facts[:10]:  # Limit to top 10
                    if goal_text.lower() in fact.object.lower():
                        similar_failures.append({
                            'fact_id': getattr(fact, 'id', None),
                            'failure_description': fact.object,
                            'timestamp': getattr(fact, 'timestamp', None),
                            'confidence': getattr(fact, 'confidence', 0.5)
                        })
        
        except Exception as e:
            logger.error(f"[ReflectiveEngine] Error finding similar failures: {e}")
        
        return similar_failures
    
    def _extract_lesson_from_failure(self, failure: Dict[str, Any]) -> Optional[str]:
        """Extract a actionable lesson from a failure record."""
        try:
            failure_desc = failure.get('failure_description', '')
            
            # Simple pattern matching for common failure types
            if 'timeout' in failure_desc.lower():
                return "Increase time estimates for similar complex operations"
            elif 'dependency' in failure_desc.lower():
                return "Validate dependencies before starting execution"
            elif 'resource' in failure_desc.lower():
                return "Check resource availability in planning phase"
            else:
                return "Add better error handling and monitoring"
        
        except Exception:
            return None
    
    def _extract_planning_memories(self) -> List[Dict[str, Any]]:
        """Extract planning-related memories for analysis."""
        planning_memories = []
        
        try:
            if hasattr(self, 'memory_log') and self.memory_log:
                # Get planning-related facts
                planning_facts = []
                
                # Search for different types of planning memories
                search_terms = [
                    ("planning_strategy", None),
                    ("plan_execution", None),
                    ("goal_achievement", None),
                    ("strategy_success", None)
                ]
                
                for subject, predicate in search_terms:
                    facts = self.memory_log.search_facts(subject=subject, predicate=predicate)
                    planning_facts.extend(facts[:20])  # Limit per type
                
                # Convert to analysis format
                for fact in planning_facts:
                    planning_memories.append({
                        'type': 'planning_fact',
                        'subject': fact.subject,
                        'predicate': fact.predicate,
                        'object': fact.object,
                        'confidence': getattr(fact, 'confidence', 0.5),
                        'timestamp': getattr(fact, 'timestamp', None)
                    })
        
        except Exception as e:
            logger.error(f"[ReflectiveEngine] Error extracting planning memories: {e}")
        
        return planning_memories
    
    def _identify_successful_planning_patterns(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Identify patterns in successful planning approaches."""
        patterns = []
        
        # Look for success indicators
        success_keywords = ['completed', 'achieved', 'successful', 'effective']
        
        for memory in memories:
            object_text = memory.get('object', '').lower()
            if any(keyword in object_text for keyword in success_keywords):
                # Extract pattern from successful case
                if 'parallel' in object_text:
                    patterns.append("Parallel execution improves success rates")
                elif 'checkpoint' in object_text:
                    patterns.append("Regular checkpoints help track progress")
                elif 'fallback' in object_text:
                    patterns.append("Fallback options increase resilience")
                elif 'incremental' in object_text:
                    patterns.append("Incremental approaches are more reliable")
        
        return list(set(patterns))  # Remove duplicates
    
    def _identify_failure_planning_patterns(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Identify patterns in planning failures."""
        patterns = []
        
        # Look for failure indicators
        failure_keywords = ['failed', 'timeout', 'error', 'blocked', 'stuck']
        
        for memory in memories:
            object_text = memory.get('object', '').lower()
            if any(keyword in object_text for keyword in failure_keywords):
                # Extract pattern from failure case
                if 'complex' in object_text:
                    patterns.append("Complex plans are more likely to fail")
                elif 'dependency' in object_text:
                    patterns.append("Dependency failures are a common cause")
                elif 'resource' in object_text:
                    patterns.append("Resource constraints often cause failures")
                elif 'time' in object_text or 'duration' in object_text:
                    patterns.append("Time estimation errors lead to failures")
        
        return list(set(patterns))  # Remove duplicates
    
    def _generate_planning_improvements(self, successful_patterns: List[str], failure_patterns: List[str]) -> List[str]:
        """Generate improvement opportunities based on pattern analysis."""
        improvements = []
        
        # Generate improvements based on failure patterns
        for pattern in failure_patterns:
            if "complex plans" in pattern.lower():
                improvements.append("Implement automatic plan complexity scoring and simplification")
            elif "dependency failures" in pattern.lower():
                improvements.append("Add comprehensive dependency validation and monitoring")
            elif "resource constraints" in pattern.lower():
                improvements.append("Implement resource planning and reservation system")
            elif "time estimation" in pattern.lower():
                improvements.append("Develop historical data-based time estimation")
        
        # Generate improvements to amplify successful patterns
        for pattern in successful_patterns:
            if "parallel execution" in pattern.lower():
                improvements.append("Increase use of parallel execution in suitable plans")
            elif "checkpoints" in pattern.lower():
                improvements.append("Standardize checkpoint implementation across all plans")
            elif "fallback options" in pattern.lower():
                improvements.append("Require fallback options for high-risk steps")
            elif "incremental approaches" in pattern.lower():
                improvements.append("Favor incremental over monolithic approaches")
        
        return improvements
    
    def _generate_planning_insights(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Generate high-level insights about planning effectiveness."""
        insights = []
        
        if len(memories) < 5:
            insights.append("Limited planning data available - need more execution history")
            return insights
        
        # Analyze confidence trends
        confidences = [m.get('confidence', 0.5) for m in memories]
        avg_confidence = sum(confidences) / len(confidences)
        
        if avg_confidence > 0.7:
            insights.append("Planning confidence is generally high - strategies are well-founded")
        elif avg_confidence < 0.4:
            insights.append("Planning confidence is low - need better strategy validation")
        
        # Analyze temporal patterns
        recent_memories = [m for m in memories if m.get('timestamp')]
        if len(recent_memories) > len(memories) * 0.8:
            insights.append("Most planning memories are recent - system is actively learning")
        
        # Analyze variety
        subjects = set(m.get('subject', '') for m in memories)
        if len(subjects) > 5:
            insights.append("Planning covers diverse domains - good generalization")
        else:
            insights.append("Planning focused on limited domains - consider broader application")
        
        return insights


def get_reflective_engine(db_path: str = "enhanced_memory.db") -> ReflectiveReplayEngine:
    """Get or create the global reflective replay engine instance."""
    if not hasattr(get_reflective_engine, '_instance'):
        get_reflective_engine._instance = ReflectiveReplayEngine(db_path)
    return get_reflective_engine._instance