#!/usr/bin/env python3
"""
Cognitive Dissonance Modeling for MeRNSTA Phase 26

Tracks internal contradiction stress, logical inconsistency, belief volatility,
and emotional-cognitive pressure across time. Simulates cognitive urgency and 
stress like a real mind under pressure.
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import math

from agents.base import BaseAgent
from config.settings import get_config


@dataclass
class DissonanceRegion:
    """Represents a region of cognitive dissonance"""
    belief_id: str
    semantic_cluster: str
    conflict_sources: List[str]
    contradiction_frequency: float
    semantic_distance: float
    causality_strength: float
    duration: float  # how long this dissonance has persisted
    pressure_score: float
    urgency: float
    confidence_erosion: float
    emotional_volatility: float
    last_updated: datetime


@dataclass
class DissonanceEvent:
    """Records a specific dissonance event"""
    timestamp: datetime
    belief_id: str
    event_type: str  # 'contradiction_detected', 'pressure_increase', 'resolution'
    source_belief: str
    target_belief: str
    intensity: float
    resolution_attempts: int = 0


class DissonanceTracker(BaseAgent):
    """
    Tracks cognitive dissonance across beliefs and memory regions.
    
    Maintains records of:
    - Contradiction frequency and semantic distance
    - Unresolved conflicts and their lifespan
    - Pressure vectors: urgency, confidence erosion, emotional volatility
    - Integration with memory systems and reflection triggers
    """
    
    def __init__(self):
        super().__init__("dissonance_tracker")
        
        # Load configuration
        self.config = get_config()
        self.dissonance_config = self.config.get('dissonance_tracking', {})
        
        # Configurable thresholds (no hardcoding per user's memory)
        self.contradiction_threshold = self.dissonance_config.get('contradiction_threshold', 0.6)
        self.pressure_threshold = self.dissonance_config.get('pressure_threshold', 0.7)
        self.urgency_threshold = self.dissonance_config.get('urgency_threshold', 0.8)
        self.volatility_threshold = self.dissonance_config.get('volatility_threshold', 0.5)
        self.resolution_timeout = self.dissonance_config.get('resolution_timeout_hours', 24)
        
        # Scoring weights (configurable)
        weights = self.dissonance_config.get('scoring_weights', {})
        self.frequency_weight = weights.get('frequency', 0.3)
        self.distance_weight = weights.get('semantic_distance', 0.25)
        self.causality_weight = weights.get('causality', 0.2)
        self.duration_weight = weights.get('duration', 0.25)
        
        # Internal state
        self.dissonance_regions: Dict[str, DissonanceRegion] = {}
        self.event_history: deque = deque(maxlen=1000)  # configurable
        self.pressure_timeline: List[Dict[str, Any]] = []
        
        # Integration hooks
        self._memory_system = None
        self._reflection_orchestrator = None
        
        # Storage
        self.storage_path = Path("output/dissonance.jsonl")
        self.storage_path.parent.mkdir(exist_ok=True)
        
        # Load persistent state
        self._load_persistent_state()
        
        logging.info(f"[{self.name}] Initialized with config: {self.dissonance_config}")
    
    @property
    def memory_system(self):
        """Lazy-load enhanced memory system for integration"""
        if self._memory_system is None:
            try:
                from storage.enhanced_memory_system import EnhancedMemorySystem
                self._memory_system = EnhancedMemorySystem()
            except ImportError as e:
                logging.error(f"[{self.name}] Could not load EnhancedMemorySystem: {e}")
                self._memory_system = None
        return self._memory_system
    
    @property
    def reflection_orchestrator(self):
        """Lazy-load reflection orchestrator for integration"""
        if self._reflection_orchestrator is None:
            try:
                from agents.reflection_orchestrator import ReflectionOrchestrator
                self._reflection_orchestrator = ReflectionOrchestrator()
            except ImportError as e:
                logging.error(f"[{self.name}] Could not load ReflectionOrchestrator: {e}")
                self._reflection_orchestrator = None
        return self._reflection_orchestrator
    
    def process_contradiction(self, contradiction_data: Dict[str, Any]) -> Optional[str]:
        """
        Process a detected contradiction and update dissonance tracking.
        
        Args:
            contradiction_data: Dict containing contradiction details
            
        Returns:
            Belief ID of the affected region, or None if not significant
        """
        try:
            # Extract contradiction details
            belief_id = contradiction_data.get('belief_id', f"belief_{int(time.time())}")
            source_belief = contradiction_data.get('source_belief', '')
            target_belief = contradiction_data.get('target_belief', '')
            semantic_distance = contradiction_data.get('semantic_distance', 0.0)
            confidence = contradiction_data.get('confidence', 0.0)
            
            # Skip low-confidence contradictions
            if confidence < self.contradiction_threshold:
                return None
            
            # Determine semantic cluster
            semantic_cluster = self._determine_semantic_cluster(source_belief, target_belief)
            
            # Update or create dissonance region
            if belief_id in self.dissonance_regions:
                region = self.dissonance_regions[belief_id]
                region.contradiction_frequency += 1.0
                region.semantic_distance = max(region.semantic_distance, semantic_distance)
                region.duration = (datetime.now() - region.last_updated).total_seconds() / 3600
                region.last_updated = datetime.now()
            else:
                region = DissonanceRegion(
                    belief_id=belief_id,
                    semantic_cluster=semantic_cluster,
                    conflict_sources=[source_belief],
                    contradiction_frequency=1.0,
                    semantic_distance=semantic_distance,
                    causality_strength=self._calculate_causality_strength(source_belief, target_belief),
                    duration=0.0,
                    pressure_score=0.0,
                    urgency=0.0,
                    confidence_erosion=0.0,
                    emotional_volatility=0.0,
                    last_updated=datetime.now()
                )
                self.dissonance_regions[belief_id] = region
            
            # Add conflict source if new
            if target_belief not in region.conflict_sources:
                region.conflict_sources.append(target_belief)
            
            # Calculate pressure vectors
            self._update_pressure_vectors(region)
            
            # Record event
            event = DissonanceEvent(
                timestamp=datetime.now(),
                belief_id=belief_id,
                event_type='contradiction_detected',
                source_belief=source_belief,
                target_belief=target_belief,
                intensity=confidence
            )
            self.event_history.append(event)
            
            # Check if reflection should be triggered
            if region.pressure_score > self.pressure_threshold:
                self._trigger_dissonance_reflection(region)
            
            # Persist state
            self._save_persistent_state()
            
            logging.info(f"[{self.name}] Processed contradiction for {belief_id}, pressure: {region.pressure_score:.3f}")
            return belief_id
            
        except Exception as e:
            logging.error(f"[{self.name}] Error processing contradiction: {e}")
            return None
    
    def _determine_semantic_cluster(self, belief1: str, belief2: str) -> str:
        """Determine semantic cluster for beliefs using simple keyword extraction"""
        # Simple clustering based on common words
        words1 = set(belief1.lower().split())
        words2 = set(belief2.lower().split())
        common_words = words1.intersection(words2)
        
        # Remove common stop words
        stop_words = {'i', 'you', 'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}
        common_words -= stop_words
        
        if common_words:
            return '_'.join(sorted(common_words)[:3])  # Use top 3 common words
        
        # Fallback to topic-based clustering
        topics = {
            'preferences': ['like', 'love', 'prefer', 'enjoy', 'hate', 'dislike'],
            'abilities': ['can', 'able', 'skilled', 'capable', 'cannot'],
            'locations': ['live', 'work', 'from', 'at', 'in'],
            'goals': ['want', 'need', 'plan', 'hope', 'intend']
        }
        
        all_words = words1.union(words2)
        for topic, keywords in topics.items():
            if any(word in all_words for word in keywords):
                return topic
        
        return 'general'
    
    def _calculate_causality_strength(self, belief1: str, belief2: str) -> float:
        """Calculate how causally linked two beliefs are"""
        # Simple heuristic based on shared concepts and causal words
        causal_words = ['because', 'since', 'due', 'caused', 'leads', 'results', 'therefore']
        
        strength = 0.0
        words1 = belief1.lower().split()
        words2 = belief2.lower().split()
        
        # Check for causal indicators
        for word in causal_words:
            if word in words1 or word in words2:
                strength += 0.2
        
        # Check for shared concepts
        shared_concepts = len(set(words1).intersection(set(words2)))
        strength += min(shared_concepts * 0.1, 0.5)
        
        return min(strength, 1.0)
    
    def _update_pressure_vectors(self, region: DissonanceRegion):
        """Update pressure vectors for a dissonance region"""
        # Calculate base dissonance score
        base_score = (
            region.contradiction_frequency * self.frequency_weight +
            region.semantic_distance * self.distance_weight +
            region.causality_strength * self.causality_weight +
            min(region.duration / 24, 1.0) * self.duration_weight  # normalize duration to 24h
        )
        
        region.pressure_score = min(base_score, 1.0)
        
        # Calculate urgency (increases non-linearly with time and frequency)
        time_factor = math.sqrt(region.duration / 24) if region.duration > 0 else 0
        frequency_factor = math.log(1 + region.contradiction_frequency) / 3
        region.urgency = min(time_factor + frequency_factor, 1.0)
        
        # Calculate confidence erosion (based on contradiction frequency and semantic distance)
        region.confidence_erosion = min(
            region.contradiction_frequency * 0.1 + region.semantic_distance * 0.3,
            1.0
        )
        
        # Calculate emotional volatility (spikes with new contradictions)
        recent_contradictions = len([e for e in self.event_history 
                                   if e.belief_id == region.belief_id and 
                                   (datetime.now() - e.timestamp).total_seconds() < 3600])
        region.emotional_volatility = min(recent_contradictions * 0.2, 1.0)
    
    def _trigger_dissonance_reflection(self, region: DissonanceRegion):
        """Trigger reflection for high-pressure dissonance regions"""
        if self.reflection_orchestrator is None:
            logging.warning(f"[{self.name}] Cannot trigger reflection - orchestrator not available")
            return
        
        try:
            context = {
                'dissonance_region': region.belief_id,
                'pressure_score': region.pressure_score,
                'conflict_sources': region.conflict_sources,
                'urgency': region.urgency,
                'trigger_source': 'dissonance_tracker'
            }
            
            # Trigger reflection through orchestrator
            from agents.reflection_orchestrator import ReflectionTrigger
            result = self.reflection_orchestrator.trigger_reflection(
                topic=f"Dissonance in {region.semantic_cluster}",
                trigger=ReflectionTrigger.CONTRADICTION,
                context=context
            )
            
            logging.info(f"[{self.name}] Triggered reflection for {region.belief_id}: {result.request_id}")
            
        except Exception as e:
            logging.error(f"[{self.name}] Failed to trigger reflection: {e}")
    
    def get_dissonance_report(self) -> Dict[str, Any]:
        """Generate comprehensive dissonance report"""
        top_regions = sorted(
            self.dissonance_regions.values(),
            key=lambda r: r.pressure_score,
            reverse=True
        )[:10]  # Top 10 most stressful regions
        
        # Calculate system-wide metrics
        total_pressure = sum(r.pressure_score for r in self.dissonance_regions.values())
        avg_pressure = total_pressure / len(self.dissonance_regions) if self.dissonance_regions else 0
        
        high_urgency_count = sum(1 for r in self.dissonance_regions.values() 
                                if r.urgency > self.urgency_threshold)
        
        # Recent activity
        recent_events = [e for e in self.event_history 
                        if (datetime.now() - e.timestamp).total_seconds() < 3600]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_regions': len(self.dissonance_regions),
                'total_pressure': total_pressure,
                'average_pressure': avg_pressure,
                'high_urgency_regions': high_urgency_count,
                'recent_events': len(recent_events)
            },
            'top_stress_regions': [
                {
                    'belief_id': r.belief_id,
                    'semantic_cluster': r.semantic_cluster,
                    'pressure_score': r.pressure_score,
                    'urgency': r.urgency,
                    'confidence_erosion': r.confidence_erosion,
                    'emotional_volatility': r.emotional_volatility,
                    'duration_hours': r.duration,
                    'contradiction_count': r.contradiction_frequency,
                    'conflict_sources': r.conflict_sources[:3]  # Top 3 sources
                }
                for r in top_regions
            ],
            'pressure_timeline': self.pressure_timeline[-20:],  # Last 20 entries
            'recent_events': [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'belief_id': e.belief_id,
                    'event_type': e.event_type,
                    'intensity': e.intensity
                }
                for e in recent_events
            ]
        }
    
    def resolve_dissonance(self, belief_id: str = None, force_evolution: bool = False) -> Dict[str, Any]:
        """
        Attempt to resolve dissonance through reflection or belief evolution.
        
        Args:
            belief_id: Specific belief to resolve, or None for highest pressure
            force_evolution: Force belief evolution vs. just reflection
            
        Returns:
            Resolution results
        """
        if belief_id is None:
            # Find highest pressure region
            if not self.dissonance_regions:
                return {'status': 'no_dissonance', 'message': 'No dissonance regions found'}
            
            belief_id = max(self.dissonance_regions.keys(), 
                          key=lambda k: self.dissonance_regions[k].pressure_score)
        
        if belief_id not in self.dissonance_regions:
            return {'status': 'error', 'message': f'Belief {belief_id} not found in dissonance regions'}
        
        region = self.dissonance_regions[belief_id]
        
        try:
            results = []
            
            if force_evolution or region.pressure_score > 0.9:
                # Attempt belief evolution for extreme dissonance
                evolution_result = self._attempt_belief_evolution(region)
                results.append(evolution_result)
            
            # Trigger targeted reflection
            reflection_result = self._trigger_targeted_reflection(region)
            results.append(reflection_result)
            
            # Update region status
            event = DissonanceEvent(
                timestamp=datetime.now(),
                belief_id=belief_id,
                event_type='resolution_attempt',
                source_belief='system',
                target_belief=region.semantic_cluster,
                intensity=region.pressure_score,
                resolution_attempts=1
            )
            self.event_history.append(event)
            
            # Reduce pressure temporarily (real resolution depends on memory system response)
            region.pressure_score *= 0.7  # 30% reduction
            region.urgency *= 0.8
            
            self._save_persistent_state()
            
            return {
                'status': 'attempted',
                'belief_id': belief_id,
                'pressure_before': region.pressure_score / 0.7,  # original
                'pressure_after': region.pressure_score,
                'results': results
            }
            
        except Exception as e:
            logging.error(f"[{self.name}] Error resolving dissonance for {belief_id}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _attempt_belief_evolution(self, region: DissonanceRegion) -> Dict[str, Any]:
        """Attempt to evolve beliefs to reduce dissonance"""
        # This would integrate with the memory system to actually modify beliefs
        # For now, just log the attempt
        logging.info(f"[{self.name}] Attempting belief evolution for {region.belief_id}")
        
        # Would use memory system to:
        # 1. Identify core conflicting beliefs
        # 2. Find compromise positions
        # 3. Update belief strengths/confidences
        # 4. Create new synthetic beliefs that resolve conflicts
        
        return {
            'type': 'belief_evolution',
            'success': False,  # Would be True if memory system integration works
            'message': 'Belief evolution not yet implemented - requires memory system integration'
        }
    
    def _trigger_targeted_reflection(self, region: DissonanceRegion) -> Dict[str, Any]:
        """Trigger targeted reflection for a specific dissonance region"""
        if self.reflection_orchestrator is None:
            return {'type': 'reflection', 'success': False, 'message': 'Reflection orchestrator not available'}
        
        try:
            from agents.reflection_orchestrator import ReflectionTrigger
            result = self.reflection_orchestrator.trigger_reflection(
                topic=f"Resolve dissonance in {region.semantic_cluster}",
                trigger=ReflectionTrigger.CONTRADICTION,
                context={
                    'dissonance_region': region.belief_id,
                    'pressure_score': region.pressure_score,
                    'conflict_sources': region.conflict_sources,
                    'resolution_request': True
                }
            )
            
            return {
                'type': 'reflection',
                'success': True,
                'reflection_id': result.request_id,
                'message': f'Triggered reflection: {result.request_id}'
            }
            
        except Exception as e:
            return {'type': 'reflection', 'success': False, 'message': str(e)}
    
    def get_dissonance_history(self, hours: int = 24) -> Dict[str, Any]:
        """Get dissonance activity history for specified time window"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_events = [
            {
                'timestamp': e.timestamp.isoformat(),
                'belief_id': e.belief_id,
                'event_type': e.event_type,
                'source_belief': e.source_belief,
                'target_belief': e.target_belief,
                'intensity': e.intensity,
                'resolution_attempts': e.resolution_attempts
            }
            for e in self.event_history 
            if e.timestamp >= cutoff_time
        ]
        
        # Group events by type
        event_counts = defaultdict(int)
        for event in recent_events:
            event_counts[event['event_type']] += 1
        
        # Calculate pressure trends
        pressure_trend = self._calculate_pressure_trend(hours)
        
        return {
            'time_window_hours': hours,
            'total_events': len(recent_events),
            'event_counts': dict(event_counts),
            'pressure_trend': pressure_trend,
            'events': recent_events
        }
    
    def _calculate_pressure_trend(self, hours: int) -> Dict[str, float]:
        """Calculate pressure trend over time window"""
        current_pressure = sum(r.pressure_score for r in self.dissonance_regions.values())
        
        # This would ideally use historical pressure data
        # For now, estimate based on event frequency
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_contradictions = len([e for e in self.event_history 
                                   if e.event_type == 'contradiction_detected' and 
                                   e.timestamp >= cutoff_time])
        
        # Simple trend calculation
        baseline_pressure = len(self.dissonance_regions) * 0.3  # estimated baseline
        pressure_change = current_pressure - baseline_pressure
        
        return {
            'current_pressure': current_pressure,
            'estimated_baseline': baseline_pressure,
            'pressure_change': pressure_change,
            'contradiction_rate': recent_contradictions / hours if hours > 0 else 0
        }
    
    def _load_persistent_state(self):
        """Load dissonance state from persistent storage"""
        try:
            if not self.storage_path.exists():
                return
            
            with open(self.storage_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get('type') == 'dissonance_region':
                        region_data = data['data']
                        region = DissonanceRegion(
                            belief_id=region_data['belief_id'],
                            semantic_cluster=region_data['semantic_cluster'],
                            conflict_sources=region_data['conflict_sources'],
                            contradiction_frequency=region_data['contradiction_frequency'],
                            semantic_distance=region_data['semantic_distance'],
                            causality_strength=region_data['causality_strength'],
                            duration=region_data['duration'],
                            pressure_score=region_data['pressure_score'],
                            urgency=region_data['urgency'],
                            confidence_erosion=region_data['confidence_erosion'],
                            emotional_volatility=region_data['emotional_volatility'],
                            last_updated=datetime.fromisoformat(region_data['last_updated'])
                        )
                        self.dissonance_regions[region.belief_id] = region
                        
                    elif data.get('type') == 'pressure_timeline':
                        self.pressure_timeline.append(data['data'])
            
            logging.info(f"[{self.name}] Loaded {len(self.dissonance_regions)} dissonance regions from storage")
            
        except Exception as e:
            logging.error(f"[{self.name}] Error loading persistent state: {e}")
    
    def _save_persistent_state(self):
        """Save dissonance state to persistent storage"""
        try:
            # Append current state
            current_time = datetime.now()
            
            with open(self.storage_path, 'a') as f:
                # Save each dissonance region
                for region in self.dissonance_regions.values():
                    region_data = {
                        'type': 'dissonance_region',
                        'timestamp': current_time.isoformat(),
                        'data': {
                            'belief_id': region.belief_id,
                            'semantic_cluster': region.semantic_cluster,
                            'conflict_sources': region.conflict_sources,
                            'contradiction_frequency': region.contradiction_frequency,
                            'semantic_distance': region.semantic_distance,
                            'causality_strength': region.causality_strength,
                            'duration': region.duration,
                            'pressure_score': region.pressure_score,
                            'urgency': region.urgency,
                            'confidence_erosion': region.confidence_erosion,
                            'emotional_volatility': region.emotional_volatility,
                            'last_updated': region.last_updated.isoformat()
                        }
                    }
                    f.write(json.dumps(region_data) + '\n')
                
                # Save pressure timeline entry
                total_pressure = sum(r.pressure_score for r in self.dissonance_regions.values())
                timeline_entry = {
                    'type': 'pressure_timeline',
                    'timestamp': current_time.isoformat(),
                    'data': {
                        'total_pressure': total_pressure,
                        'region_count': len(self.dissonance_regions),
                        'avg_pressure': total_pressure / len(self.dissonance_regions) if self.dissonance_regions else 0
                    }
                }
                f.write(json.dumps(timeline_entry) + '\n')
                self.pressure_timeline.append(timeline_entry['data'])
            
        except Exception as e:
            logging.error(f"[{self.name}] Error saving persistent state: {e}")
    
    def integrate_with_memory_system(self):
        """Hook into memory system for contradiction detection"""
        if self.memory_system is None:
            logging.warning(f"[{self.name}] Cannot integrate - memory system not available")
            return False
        
        try:
            # This would hook into the memory system's contradiction detection
            # For now, just log the integration attempt
            logging.info(f"[{self.name}] Integrating with enhanced memory system")
            
            # The memory system would call our process_contradiction method
            # when contradictions are detected
            
            return True
            
        except Exception as e:
            logging.error(f"[{self.name}] Failed to integrate with memory system: {e}")
            return False
    
    def cleanup_old_regions(self, max_age_hours: int = None):
        """Clean up old dissonance regions that have been resolved"""
        if max_age_hours is None:
            max_age_hours = self.dissonance_config.get('cleanup_age_hours', 168)  # 1 week default
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        regions_to_remove = []
        for belief_id, region in self.dissonance_regions.items():
            # Remove if old and low pressure
            if (region.last_updated < cutoff_time and 
                region.pressure_score < 0.1 and
                region.urgency < 0.1):
                regions_to_remove.append(belief_id)
        
        for belief_id in regions_to_remove:
            del self.dissonance_regions[belief_id]
        
        if regions_to_remove:
            logging.info(f"[{self.name}] Cleaned up {len(regions_to_remove)} old dissonance regions")
            self._save_persistent_state()
    
    # Abstract method implementations required by BaseAgent
    
    def get_agent_instructions(self) -> str:
        """Get instructions for the DissonanceTracker agent"""
        return """
        You are the DissonanceTracker agent. Your role is to:
        
        1. Monitor cognitive dissonance across the belief system
        2. Track contradiction frequency, semantic distance, and causality strength
        3. Calculate pressure vectors: urgency, confidence erosion, emotional volatility
        4. Trigger reflections when dissonance pressure exceeds thresholds
        5. Maintain persistent records of dissonance patterns and resolution attempts
        
        Key capabilities:
        - Process contradiction data from memory systems
        - Generate comprehensive dissonance reports  
        - Resolve high-pressure dissonance through reflection triggers
        - Track temporal patterns in belief conflicts
        - Integrate with reflection orchestrator for targeted resolution
        
        Always prioritize unresolved high-pressure contradictions and maintain
        accurate tracking of dissonance evolution over time.
        """
    
    def respond(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Respond to queries about dissonance state and provide analysis.
        
        Args:
            query: Query about dissonance tracking
            context: Additional context for the query
            
        Returns:
            Response string with dissonance analysis
        """
        query_lower = query.lower()
        
        try:
            if 'report' in query_lower or 'status' in query_lower:
                report = self.get_dissonance_report()
                summary = report['summary']
                response = f"🧠 Dissonance Status: {summary['total_regions']} regions, "
                response += f"pressure: {summary['total_pressure']:.3f}, "
                response += f"urgent: {summary['high_urgency_regions']}"
                
                if report['top_stress_regions']:
                    top_region = report['top_stress_regions'][0]
                    response += f"\nTop stress: {top_region['belief_id']} "
                    response += f"(pressure: {top_region['pressure_score']:.3f})"
                
                return response
                
            elif 'resolve' in query_lower:
                # Extract belief_id if specified
                belief_id = None
                if context and 'belief_id' in context:
                    belief_id = context['belief_id']
                
                result = self.resolve_dissonance(belief_id)
                if result['status'] == 'attempted':
                    return f"Resolution attempted for {result['belief_id']}: pressure reduced by {((result['pressure_before'] - result['pressure_after']) / result['pressure_before'] * 100):.1f}%"
                else:
                    return f"Resolution failed: {result.get('message', 'Unknown error')}"
                    
            elif 'history' in query_lower:
                hours = 24
                if context and 'hours' in context:
                    hours = int(context.get('hours', 24))
                
                history = self.get_dissonance_history(hours)
                return f"Dissonance history ({hours}h): {history['total_events']} events, current pressure trend: {history['pressure_trend']['pressure_change']:+.3f}"
                
            elif 'pressure' in query_lower:
                total_pressure = sum(r.pressure_score for r in self.dissonance_regions.values())
                high_pressure_count = sum(1 for r in self.dissonance_regions.values() if r.pressure_score > 0.7)
                return f"Total system pressure: {total_pressure:.3f}, high-pressure regions: {high_pressure_count}"
                
            elif 'clean' in query_lower or 'cleanup' in query_lower:
                old_count = len(self.dissonance_regions)
                self.cleanup_old_regions()
                new_count = len(self.dissonance_regions)
                cleaned = old_count - new_count
                return f"Cleanup complete: removed {cleaned} old regions, {new_count} active regions remain"
                
            else:
                # General status
                total_regions = len(self.dissonance_regions)
                total_pressure = sum(r.pressure_score for r in self.dissonance_regions.values())
                return f"DissonanceTracker: monitoring {total_regions} regions with total pressure {total_pressure:.3f}. Try 'report', 'resolve', 'history', or 'cleanup'."
                
        except Exception as e:
            logging.error(f"[{self.name}] Error in respond: {e}")
            return f"Error processing query: {str(e)}"


# Singleton instance for global access
_dissonance_tracker_instance = None


def get_dissonance_tracker() -> DissonanceTracker:
    """Get the global DissonanceTracker instance"""
    global _dissonance_tracker_instance
    if _dissonance_tracker_instance is None:
        _dissonance_tracker_instance = DissonanceTracker()
    return _dissonance_tracker_instance