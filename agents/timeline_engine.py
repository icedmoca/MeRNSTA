#!/usr/bin/env python3
"""
Timeline Engine for MeRNSTA Phase 24: World-State Temporal Timeline & Event Inference

This module provides a temporal layer to the world model, enabling the system to:
- Track the evolution of beliefs over time
- Infer timelines and reason about sequences of events
- Detect temporal inconsistencies and causal anomalies
- Store and query timestamped belief updates
"""

import logging
import time
import json
import os
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import math
from enum import Enum

from .base import BaseAgent
from config.settings import get_config


class EventType(Enum):
    """Types of temporal events that can be tracked"""
    OBSERVATION = "observation"
    INFERRED = "inferred"
    CONTRADICTION = "contradiction"
    PREDICTION = "prediction"
    UPDATE = "update"
    BELIEF_CHANGE = "belief_change"


@dataclass
class TemporalEvent:
    """Represents a timestamped event in the world model timeline"""
    timestamp: datetime
    fact: str
    confidence: float
    source: str
    event_type: EventType
    reasoning_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate unique event ID if not provided"""
        if self.event_id is None:
            timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S_%f")
            self.event_id = f"event_{timestamp_str}_{hash(self.fact) % 10000:04d}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'fact': self.fact,
            'confidence': self.confidence,
            'source': self.source,
            'event_type': self.event_type.value,
            'reasoning_agent': self.reasoning_agent,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemporalEvent':
        """Create TemporalEvent from dictionary"""
        return cls(
            event_id=data['event_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            fact=data['fact'],
            confidence=data['confidence'],
            source=data['source'],
            event_type=EventType(data['event_type']),
            reasoning_agent=data.get('reasoning_agent'),
            metadata=data.get('metadata', {})
        )


@dataclass
class CausalSequence:
    """Represents a sequence of causally-linked events"""
    events: List[TemporalEvent]
    causal_strength: float
    confidence: float
    sequence_id: str
    detected_at: datetime = field(default_factory=datetime.now)
    
    def is_temporally_consistent(self) -> bool:
        """Check if events in sequence occur in chronological order"""
        for i in range(len(self.events) - 1):
            if self.events[i].timestamp > self.events[i + 1].timestamp:
                return False
        return True


class TemporalTimeline(BaseAgent):
    """
    Manages the temporal timeline of world state changes and belief evolution.
    
    Tracks timestamped belief updates, represents belief changes as temporal events,
    and provides methods for temporal reasoning and anomaly detection.
    """
    
    def __init__(self, agent_id: str = "timeline_engine", 
                 persistence_path: str = "output/world_timeline.jsonl",
                 max_events: int = 10000):
        # Initialize parent class first
        super().__init__(agent_id)
        
        # Set agent_id for consistency (BaseAgent uses 'name')
        self.agent_id = agent_id
        
        self.persistence_path = persistence_path
        self.max_events = max_events
        
        # Core timeline storage
        self.events: List[TemporalEvent] = []
        self.events_by_subject: Dict[str, List[TemporalEvent]] = defaultdict(list)
        self.events_by_type: Dict[EventType, List[TemporalEvent]] = defaultdict(list)
        self.events_by_source: Dict[str, List[TemporalEvent]] = defaultdict(list)
        
        # Causal sequences
        self.causal_sequences: List[CausalSequence] = []
        
        # Temporal anomalies tracking
        self.anomalies: List[Dict[str, Any]] = []
        
        # Configuration
        self.config = get_config()
        self.temporal_window_hours = self.config.get('timeline_temporal_window_hours', 168)  # 1 week default
        self.causality_threshold = self.config.get('timeline_causality_threshold', 0.7)
        self.anomaly_detection_enabled = self.config.get('timeline_anomaly_detection', True)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        
        # Load existing timeline if available
        self._load_timeline()
        
        logging.info(f"[{self.agent_id}] Timeline engine initialized with {len(self.events)} events")
    
    def get_agent_instructions(self) -> str:
        """Get instructions for this agent's role and capabilities"""
        return """You are the Timeline Engine agent for MeRNSTA's Phase 24 World-State Temporal Timeline & Event Inference system.

Your primary responsibilities are:
1. Track the evolution of beliefs and world state over time
2. Store timestamped events representing belief updates, observations, and inferences
3. Provide temporal querying capabilities to find events about specific subjects
4. Detect causal sequences and temporal relationships between events
5. Identify temporal inconsistencies and logical anomalies
6. Generate timeline summaries and statistics

Key capabilities:
- Event storage with rich metadata and temporal indexing
- Subject-based event querying with time window filtering
- Causal sequence detection and validation
- Temporal anomaly detection (effects before causes, contradictions)
- Timeline summarization with temporal pattern analysis
- Persistent storage of timeline data in JSON Lines format

Use your timeline capabilities to help track how beliefs change over time,
detect causal relationships, and identify temporal inconsistencies in the world model."""
    
    def respond(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a message and respond using timeline capabilities"""
        
        message_lower = message.lower()
        
        try:
            # Handle different types of timeline queries
            if 'events about' in message_lower or 'timeline for' in message_lower:
                # Extract subject from message
                subject = message.replace('events about', '').replace('timeline for', '').strip()
                events = self.get_events_about(subject)
                
                response = f"Found {len(events)} events about '{subject}'"
                if events:
                    recent_events = events[-5:]  # Last 5 events
                    response += ":\n"
                    for event in recent_events:
                        response += f"- {event.timestamp.strftime('%Y-%m-%d %H:%M')}: {event.fact} (confidence: {event.confidence:.2f})\n"
                else:
                    response += ". Try a broader search term or check recent activity."
                
                return {
                    'response': response,
                    'events_found': len(events),
                    'agent': 'timeline_engine'
                }
                
            elif 'sequence' in message_lower and '->' in message:
                # Parse causal sequence query
                sequence_part = message.split('sequence')[-1].strip()
                if '->' in sequence_part:
                    causal_chain = [event.strip() for event in sequence_part.split('->')]
                    sequence = self.get_event_sequence(causal_chain)
                    
                    if sequence:
                        response = f"Found causal sequence: {' → '.join(causal_chain)}\n"
                        response += f"Causal strength: {sequence.causal_strength:.2f}\n"
                        response += f"Confidence: {sequence.confidence:.2f}\n"
                        response += f"Temporally consistent: {'Yes' if sequence.is_temporally_consistent() else 'No'}"
                    else:
                        response = f"No causal sequence found for: {' → '.join(causal_chain)}"
                    
                    return {
                        'response': response,
                        'sequence_found': sequence is not None,
                        'agent': 'timeline_engine'
                    }
            
            elif 'anomalies' in message_lower or 'inconsistencies' in message_lower:
                # Detect temporal anomalies
                anomalies = self.detect_temporal_inconsistencies()
                
                response = f"Found {len(anomalies)} temporal anomalies"
                if anomalies:
                    response += ":\n"
                    for anomaly in anomalies[-3:]:  # Last 3 anomalies
                        response += f"- {anomaly['type']}: {anomaly['description']}\n"
                else:
                    response += ". Timeline appears temporally consistent."
                
                return {
                    'response': response,
                    'anomalies_found': len(anomalies),
                    'agent': 'timeline_engine'
                }
            
            elif 'statistics' in message_lower or 'stats' in message_lower or 'summary' in message_lower:
                # Get timeline statistics
                stats = self.get_timeline_stats()
                
                response = f"Timeline Statistics:\n"
                response += f"- Total events: {stats['total_events']}\n"
                response += f"- Timeline span: {stats['timeline_span_hours']:.1f} hours\n"
                response += f"- Average confidence: {stats['average_confidence']}\n"
                response += f"- Causal sequences: {stats['causal_sequences']}\n"
                response += f"- Detected anomalies: {stats['detected_anomalies']}"
                
                return {
                    'response': response,
                    'statistics': stats,
                    'agent': 'timeline_engine'
                }
            
            else:
                # General help response
                return {
                    'response': "I'm the Timeline Engine. I can help you:\n" +
                               "- Track events: 'events about <subject>'\n" +
                               "- Find sequences: 'sequence <event1> -> <event2>'\n" +
                               "- Detect anomalies: 'show anomalies'\n" +
                               "- Get statistics: 'timeline stats'",
                    'agent': 'timeline_engine'
                }
                
        except Exception as e:
            logging.error(f"[{self.agent_id}] Error processing message: {e}")
            return {
                'response': f"I encountered an error while processing your request: {str(e)}",
                'error': str(e),
                'agent': 'timeline_engine'
            }
    
    def add_event(self, fact: str, confidence: float, timestamp: Optional[datetime] = None,
                  source: str = "unknown", event_type: EventType = EventType.OBSERVATION,
                  reasoning_agent: Optional[str] = None, metadata: Optional[Dict] = None) -> TemporalEvent:
        """
        Add a new event to the timeline.
        
        Args:
            fact: The belief or observation being recorded
            confidence: Confidence score (0.0 to 1.0)
            timestamp: When the event occurred (defaults to now)
            source: Source of the event
            event_type: Type of event being recorded
            reasoning_agent: Which agent produced this event
            metadata: Additional event metadata
        
        Returns:
            The created TemporalEvent
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if metadata is None:
            metadata = {}
        
        # Validate and normalize confidence value
        confidence = max(0.0, min(1.0, confidence))
        
        # Create the event
        event = TemporalEvent(
            timestamp=timestamp,
            fact=fact,
            confidence=confidence,
            source=source,
            event_type=event_type,
            reasoning_agent=reasoning_agent,
            metadata=metadata
        )
        
        # Add to main timeline
        self.events.append(event)
        
        # Update indices
        self._update_indices(event)
        
        # Check for causal relationships and anomalies
        if self.anomaly_detection_enabled:
            self._detect_temporal_anomalies(event)
            self._detect_causal_sequences(event)
        
        # Maintain timeline size
        self._manage_timeline_capacity()
        
        # Persist changes
        self._persist_event(event)
        
        logging.debug(f"[{self.agent_id}] Added event: {fact[:50]}... (confidence: {confidence})")
        
        return event
    
    def get_events_about(self, subject: str, time_window_hours: Optional[int] = None) -> List[TemporalEvent]:
        """
        Get all events related to a specific subject.
        
        Args:
            subject: The subject to search for
            time_window_hours: Optional time window to limit search
        
        Returns:
            List of relevant TemporalEvents, sorted by timestamp
        """
        # Find events that mention the subject
        relevant_events = []
        
        # Check both exact matches and partial matches
        subject_lower = subject.lower()
        
        for event in self.events:
            if (subject_lower in event.fact.lower() or 
                subject_lower in event.source.lower() or
                any(subject_lower in str(v).lower() for v in event.metadata.values())):
                
                # Apply time window filter if specified
                if time_window_hours is not None:
                    cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
                    if event.timestamp < cutoff_time:
                        continue
                
                relevant_events.append(event)
        
        # Sort by timestamp
        relevant_events.sort(key=lambda e: e.timestamp)
        
        logging.debug(f"[{self.agent_id}] Found {len(relevant_events)} events about '{subject}'")
        
        return relevant_events
    
    def get_event_sequence(self, causal_chain: List[str]) -> Optional[CausalSequence]:
        """
        Search for a sequence of events matching a causal chain.
        
        Args:
            causal_chain: List of facts/concepts that should occur in sequence
        
        Returns:
            CausalSequence if found, None otherwise
        """
        if len(causal_chain) < 2:
            return None
        
        # Find events for each element in the chain
        chain_events = []
        for concept in causal_chain:
            concept_events = self.get_events_about(concept)
            if not concept_events:
                return None  # Missing link in chain
            chain_events.append(concept_events)
        
        # Try to find a valid temporal sequence
        best_sequence = None
        best_score = 0.0
        
        # Generate all possible combinations
        import itertools
        for event_combination in itertools.product(*chain_events):
            # Check temporal ordering
            if self._is_temporally_ordered(event_combination):
                # Calculate causal strength
                strength = self._calculate_causal_strength(event_combination)
                confidence = min(e.confidence for e in event_combination)
                
                if strength > best_score:
                    best_score = strength
                    sequence_id = f"seq_{int(time.time())}_{hash('_'.join(causal_chain)) % 1000:03d}"
                    best_sequence = CausalSequence(
                        events=list(event_combination),
                        causal_strength=strength,
                        confidence=confidence,
                        sequence_id=sequence_id
                    )
        
        if best_sequence and best_score >= self.causality_threshold:
            logging.info(f"[{self.agent_id}] Found causal sequence: {' -> '.join(causal_chain)} (strength: {best_score:.2f})")
            return best_sequence
        
        return None
    
    def summarize_timeline(self, window: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """
        Provide a summary of events within a time window.
        
        Args:
            window: Tuple of (start_time, end_time)
        
        Returns:
            Dictionary containing timeline summary
        """
        start_time, end_time = window
        
        # Filter events in window
        window_events = [
            event for event in self.events
            if start_time <= event.timestamp <= end_time
        ]
        
        # Group by event type
        events_by_type = defaultdict(list)
        for event in window_events:
            events_by_type[event.event_type].append(event)
        
        # Calculate statistics
        total_events = len(window_events)
        avg_confidence = sum(e.confidence for e in window_events) / max(1, total_events)
        
        # Find most active subjects
        subject_counts = defaultdict(int)
        for event in window_events:
            # Extract subjects from facts (simple approach)
            words = event.fact.split()
            for word in words[:3]:  # Consider first few words as potential subjects
                if len(word) > 2:
                    subject_counts[word.lower()] += 1
        
        top_subjects = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Detect patterns
        temporal_patterns = self._analyze_temporal_patterns(window_events)
        
        summary = {
            'time_window': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration_hours': (end_time - start_time).total_seconds() / 3600
            },
            'statistics': {
                'total_events': total_events,
                'average_confidence': round(avg_confidence, 3),
                'events_by_type': {
                    event_type.value: len(events) 
                    for event_type, events in events_by_type.items()
                },
                'unique_sources': len(set(e.source for e in window_events))
            },
            'top_subjects': top_subjects,
            'temporal_patterns': temporal_patterns,
            'causal_sequences_detected': len([
                seq for seq in self.causal_sequences 
                if any(start_time <= e.timestamp <= end_time for e in seq.events)
            ]),
            'anomalies_detected': len([
                anomaly for anomaly in self.anomalies
                if 'timestamp' in anomaly and 
                start_time <= datetime.fromisoformat(anomaly['timestamp']) <= end_time
            ])
        }
        
        logging.info(f"[{self.agent_id}] Generated timeline summary for {total_events} events")
        
        return summary
    
    def detect_temporal_inconsistencies(self) -> List[Dict[str, Any]]:
        """
        Detect temporal inconsistencies and logical anomalies in the timeline.
        
        Returns:
            List of detected anomalies with details
        """
        inconsistencies = []
        
        # Check causal sequences for temporal violations
        for sequence in self.causal_sequences:
            if not sequence.is_temporally_consistent():
                inconsistencies.append({
                    'type': 'temporal_violation',
                    'description': 'Effect occurs before cause',
                    'sequence_id': sequence.sequence_id,
                    'events': [e.fact for e in sequence.events],
                    'timestamps': [e.timestamp.isoformat() for e in sequence.events],
                    'detected_at': datetime.now().isoformat()
                })
        
        # Check for contradictory beliefs about the same concept
        contradictions = self._find_contradictory_beliefs()
        for contradiction in contradictions:
            inconsistencies.append({
                'type': 'belief_contradiction',
                'description': 'Contradictory beliefs about the same concept',
                'concept': contradiction['concept'],
                'conflicting_events': contradiction['events'],
                'detected_at': datetime.now().isoformat()
            })
        
        # Check for impossible time sequences
        impossible_sequences = self._find_impossible_sequences()
        for sequence in impossible_sequences:
            inconsistencies.append({
                'type': 'impossible_sequence',
                'description': 'Events occur in physically impossible order',
                'events': sequence['events'],
                'reason': sequence['reason'],
                'detected_at': datetime.now().isoformat()
            })
        
        # Update anomalies list
        self.anomalies.extend(inconsistencies)
        
        logging.info(f"[{self.agent_id}] Detected {len(inconsistencies)} temporal inconsistencies")
        
        return inconsistencies
    
    def get_timeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive timeline statistics"""
        now = datetime.now()
        
        # Time-based statistics
        if self.events:
            oldest_event = min(self.events, key=lambda e: e.timestamp)
            newest_event = max(self.events, key=lambda e: e.timestamp)
            timeline_span = (newest_event.timestamp - oldest_event.timestamp).total_seconds() / 3600
        else:
            timeline_span = 0
            oldest_event = newest_event = None
        
        # Event type distribution
        type_distribution = {}
        for event_type in EventType:
            type_distribution[event_type.value] = len(self.events_by_type[event_type])
        
        # Source distribution
        source_distribution = {}
        for source, events in self.events_by_source.items():
            source_distribution[source] = len(events)
        
        return {
            'total_events': len(self.events),
            'timeline_span_hours': round(timeline_span, 2),
            'oldest_event': oldest_event.timestamp.isoformat() if oldest_event else None,
            'newest_event': newest_event.timestamp.isoformat() if newest_event else None,
            'event_type_distribution': type_distribution,
            'source_distribution': source_distribution,
            'causal_sequences': len(self.causal_sequences),
            'detected_anomalies': len(self.anomalies),
            'unique_subjects': len(self.events_by_subject),
            'average_confidence': round(
                sum(e.confidence for e in self.events) / max(1, len(self.events)), 3
            )
        }
    
    # Private helper methods
    
    def _update_indices(self, event: TemporalEvent):
        """Update search indices with new event"""
        # Index by subject (extract subjects from fact)
        words = event.fact.lower().split()
        for word in words:
            if len(word) > 2:  # Skip short words
                self.events_by_subject[word].append(event)
        
        # Index by type and source
        self.events_by_type[event.event_type].append(event)
        self.events_by_source[event.source].append(event)
    
    def _detect_temporal_anomalies(self, new_event: TemporalEvent):
        """Detect anomalies involving the new event"""
        # Look for events that might be causally related but temporally inconsistent
        recent_events = [
            e for e in self.events[-100:]  # Check last 100 events
            if e != new_event and 
            abs((e.timestamp - new_event.timestamp).total_seconds()) < 3600  # Within 1 hour
        ]
        
        for event in recent_events:
            # Simple causal relationship detection
            if self._might_be_causally_related(event, new_event):
                if event.timestamp > new_event.timestamp:
                    # Potential temporal violation
                    anomaly = {
                        'type': 'potential_temporal_violation',
                        'cause_event': new_event.fact,
                        'effect_event': event.fact,
                        'cause_time': new_event.timestamp.isoformat(),
                        'effect_time': event.timestamp.isoformat(),
                        'detected_at': datetime.now().isoformat()
                    }
                    self.anomalies.append(anomaly)
    
    def _detect_causal_sequences(self, new_event: TemporalEvent):
        """Look for causal sequences involving the new event"""
        # This is a simplified implementation
        # In a full system, this would use more sophisticated NLP and causal reasoning
        
        # Look for events that might be part of a causal sequence
        recent_events = [
            e for e in self.events[-50:]  # Check last 50 events
            if e != new_event and 
            abs((e.timestamp - new_event.timestamp).total_seconds()) < 7200  # Within 2 hours
        ]
        
        for event in recent_events:
            if self._might_be_causally_related(event, new_event):
                # Create a simple causal sequence
                sequence_id = f"auto_seq_{int(time.time())}_{len(self.causal_sequences)}"
                sequence = CausalSequence(
                    events=[event, new_event] if event.timestamp < new_event.timestamp else [new_event, event],
                    causal_strength=0.6,  # Default moderate strength
                    confidence=min(event.confidence, new_event.confidence),
                    sequence_id=sequence_id
                )
                self.causal_sequences.append(sequence)
    
    def _might_be_causally_related(self, event1: TemporalEvent, event2: TemporalEvent) -> bool:
        """Simple heuristic to determine if two events might be causally related"""
        # Look for common words, causal indicators, etc.
        fact1_words = set(event1.fact.lower().split())
        fact2_words = set(event2.fact.lower().split())
        
        # Check for shared concepts
        shared_words = fact1_words.intersection(fact2_words)
        if len(shared_words) >= 2:
            return True
        
        # Check for causal indicators
        causal_indicators = ['because', 'due to', 'caused', 'leads to', 'results in', 'then', 'so']
        for indicator in causal_indicators:
            if indicator in event1.fact.lower() or indicator in event2.fact.lower():
                return True
        
        return False
    
    def _is_temporally_ordered(self, events: List[TemporalEvent]) -> bool:
        """Check if events are in chronological order"""
        for i in range(len(events) - 1):
            if events[i].timestamp > events[i + 1].timestamp:
                return False
        return True
    
    def _calculate_causal_strength(self, events: List[TemporalEvent]) -> float:
        """Calculate causal strength for a sequence of events"""
        if len(events) < 2:
            return 0.0
        
        # Base strength on temporal proximity and confidence
        total_strength = 0.0
        for i in range(len(events) - 1):
            time_diff = (events[i + 1].timestamp - events[i].timestamp).total_seconds()
            
            # Closer in time = stronger causal relationship
            proximity_factor = math.exp(-time_diff / 3600)  # Decay over hours
            confidence_factor = (events[i].confidence + events[i + 1].confidence) / 2
            
            total_strength += proximity_factor * confidence_factor
        
        return total_strength / (len(events) - 1)
    
    def _analyze_temporal_patterns(self, events: List[TemporalEvent]) -> Dict[str, Any]:
        """Analyze temporal patterns in events"""
        if not events:
            return {}
        
        # Group by hour of day
        hourly_distribution = defaultdict(int)
        for event in events:
            hour = event.timestamp.hour
            hourly_distribution[hour] += 1
        
        # Group by day of week
        daily_distribution = defaultdict(int)
        for event in events:
            day = event.timestamp.strftime('%A')
            daily_distribution[day] += 1
        
        # Find peak activity periods
        peak_hour = max(hourly_distribution.items(), key=lambda x: x[1]) if hourly_distribution else (0, 0)
        peak_day = max(daily_distribution.items(), key=lambda x: x[1]) if daily_distribution else ("", 0)
        
        return {
            'hourly_distribution': dict(hourly_distribution),
            'daily_distribution': dict(daily_distribution),
            'peak_hour': {'hour': peak_hour[0], 'count': peak_hour[1]},
            'peak_day': {'day': peak_day[0], 'count': peak_day[1]},
            'event_rate_per_hour': len(events) / max(1, 
                (max(e.timestamp for e in events) - min(e.timestamp for e in events)).total_seconds() / 3600
            ) if len(events) > 1 else 0
        }
    
    def _find_contradictory_beliefs(self) -> List[Dict[str, Any]]:
        """Find contradictory beliefs about the same concepts"""
        contradictions = []
        
        # Group events by extracted concepts
        concept_events = defaultdict(list)
        for event in self.events:
            # Simple concept extraction
            words = event.fact.lower().split()
            for word in words:
                if len(word) > 3:  # Focus on longer words
                    concept_events[word].append(event)
        
        # Look for contradictions
        for concept, events in concept_events.items():
            if len(events) >= 2:
                # Look for opposing statements
                positive_events = [e for e in events if 'not' not in e.fact.lower() and 'never' not in e.fact.lower()]
                negative_events = [e for e in events if 'not' in e.fact.lower() or 'never' in e.fact.lower()]
                
                if positive_events and negative_events:
                    contradictions.append({
                        'concept': concept,
                        'events': [e.fact for e in positive_events + negative_events]
                    })
        
        return contradictions
    
    def _find_impossible_sequences(self) -> List[Dict[str, Any]]:
        """Find sequences that are physically or logically impossible"""
        impossible = []
        
        # This is a simplified implementation
        # In practice, this would involve more sophisticated reasoning
        
        for sequence in self.causal_sequences:
            events = sequence.events
            
            # Check for events happening too close together to be physically possible
            for i in range(len(events) - 1):
                time_diff = (events[i + 1].timestamp - events[i].timestamp).total_seconds()
                if time_diff < 0.1:  # Less than 0.1 seconds apart
                    impossible.append({
                        'events': [e.fact for e in events],
                        'reason': 'Events too close in time to be physically possible'
                    })
                    break
        
        return impossible
    
    def _manage_timeline_capacity(self):
        """Manage timeline size to prevent unbounded growth"""
        if len(self.events) > self.max_events:
            # Remove oldest events, keeping indices updated
            events_to_remove = len(self.events) - self.max_events
            removed_events = self.events[:events_to_remove]
            self.events = self.events[events_to_remove:]
            
            # Update indices
            for event in removed_events:
                # Remove from subject indices
                words = event.fact.lower().split()
                for word in words:
                    if word in self.events_by_subject:
                        if event in self.events_by_subject[word]:
                            self.events_by_subject[word].remove(event)
                
                # Remove from type and source indices
                if event in self.events_by_type[event.event_type]:
                    self.events_by_type[event.event_type].remove(event)
                if event in self.events_by_source[event.source]:
                    self.events_by_source[event.source].remove(event)
            
            logging.info(f"[{self.agent_id}] Removed {events_to_remove} old events to maintain capacity")
    
    def _persist_event(self, event: TemporalEvent):
        """Persist a single event to storage"""
        try:
            with open(self.persistence_path, 'a', encoding='utf-8') as f:
                json.dump(event.to_dict(), f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            logging.error(f"[{self.agent_id}] Failed to persist event: {e}")
    
    def _load_timeline(self):
        """Load existing timeline from storage"""
        if not os.path.exists(self.persistence_path):
            return
        
        try:
            with open(self.persistence_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        event = TemporalEvent.from_dict(data)
                        self.events.append(event)
                        self._update_indices(event)
            
            # Sort events by timestamp
            self.events.sort(key=lambda e: e.timestamp)
            
            logging.info(f"[{self.agent_id}] Loaded {len(self.events)} events from {self.persistence_path}")
            
        except Exception as e:
            logging.error(f"[{self.agent_id}] Failed to load timeline: {e}")


# Global timeline engine instance
_timeline_engine = None


def get_timeline_engine() -> TemporalTimeline:
    """Get the global timeline engine instance"""
    global _timeline_engine
    if _timeline_engine is None:
        _timeline_engine = TemporalTimeline()
    return _timeline_engine


def reset_timeline_engine():
    """Reset the global timeline engine (useful for testing)"""
    global _timeline_engine
    _timeline_engine = None