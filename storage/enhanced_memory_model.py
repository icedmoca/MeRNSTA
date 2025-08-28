"""
Enhanced Memory Model for MeRNSTA
Includes confidence, contradiction tracking, volatility, and temporal support
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import time
import uuid
import math
from collections import Counter


@dataclass
class EnhancedTripletFact:
    """
    Enhanced fact representation with Phase 2 autonomous cognitive capabilities:
    - Causal & Temporal Linkage
    - Theory of Mind perspective support
    - Advanced contradiction and volatility tracking
    - Token-level entropy and causal propagation tracking
    """
    # Core triplet
    subject: str
    predicate: str
    object: str
    
    # Temporal and confidence
    timestamp: Optional[float] = None  # Unix timestamp for consistency
    confidence: float = 0.8
    
    # Contradiction and volatility tracking
    contradiction: bool = False
    contradicts_with: List[str] = field(default_factory=list)  # UUIDs of contradicting facts
    volatile: bool = False
    volatility_score: float = 0.0
    
    # Additional metadata
    id: Optional[str] = None  # Use UUID strings for better compatibility
    user_profile_id: Optional[str] = None
    session_id: Optional[str] = None
    source_message_id: Optional[int] = None
    
    # Linguistic metadata
    hedge_detected: bool = False  # "I guess", "probably"
    intensifier_detected: bool = False  # "absolutely", "definitely"
    negation: bool = False
    
    # Embedding for semantic search
    embedding: Optional[List[float]] = None
    
    # PHASE 2: Causal & Temporal Linkage
    cause: Optional[str] = None  # String description or fact ID pointing to causal origin
    preceded_by: Optional[str] = None  # Link to previous belief evolution (fact ID)
    causal_strength: float = 0.0  # Strength of causal relationship (0.0-1.0)
    
    # PHASE 2: Theory of Mind - Perspective Support
    perspective: str = "user"  # Who holds this belief ("user", "system", person name)
    source: Optional[str] = None  # Who reported this belief
    confidence_by_subject: Dict[str, float] = field(default_factory=dict)  # Confidence per perspective
    nested_belief: bool = False  # True for "user believes that Anna believes..."
    belief_about: Optional[str] = None  # For nested beliefs: who this belief is about
    
    # TOKEN-LEVEL ENTROPY & CAUSAL TRACKING
    token_count: Optional[int] = None  # Number of tokens in the fact
    token_entropy: Optional[float] = None  # Entropy of token distribution
    token_ids: Optional[List[int]] = field(default_factory=list)  # Actual token IDs
    token_hash: Optional[str] = None  # Hash of token sequence for quick comparison
    causal_token_propagation: Dict[str, List[int]] = field(default_factory=dict)  # Token→fact graph
    
    # Memory management
    active: bool = True
    last_accessed: Optional[float] = None
    access_count: int = 0
    
    # History tracking
    change_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # PHASE 8: Emotional Context and Tagging
    emotion_valence: Optional[float] = None  # Emotional valence (-1.0 to 1.0)
    emotion_arousal: Optional[float] = None  # Emotional arousal (0.0 to 1.0)
    emotion_tag: Optional[str] = None  # Emotion label (e.g., "curiosity", "frustration")
    emotional_strength: float = 0.0  # Strength of emotional association (0.0 to 1.0)
    emotion_source: Optional[str] = None  # Source of emotional tagging
    mood_context: Optional[str] = None  # Mood state when fact was created/modified
    
    # PHASE 10: Life Narrative Integration
    episode_id: Optional[str] = None  # ID of the life narrative episode this fact belongs to
    
    def __post_init__(self):
        """Initialize computed fields after creation."""
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.last_accessed is None:
            self.last_accessed = self.timestamp
        if self.id is None:
            self.id = str(uuid.uuid4())
        
        # Initialize confidence_by_subject with primary perspective
        if not self.confidence_by_subject:
            self.confidence_by_subject[self.perspective] = self.confidence
    
    def add_causal_link(self, cause_description: str, strength: float = 0.8):
        """Add a causal link to this fact."""
        self.cause = cause_description
        self.causal_strength = strength
        self.change_history.append({
            "timestamp": time.time(),
            "action": "causal_link_added",
            "cause": cause_description,
            "strength": strength
        })
    
    def add_temporal_link(self, preceded_by_fact_id: str):
        """Link this fact to a previous belief evolution."""
        self.preceded_by = preceded_by_fact_id
        self.change_history.append({
            "timestamp": time.time(),
            "action": "temporal_link_added",
            "preceded_by": preceded_by_fact_id
        })
    
    def add_perspective(self, perspective: str, confidence: float):
        """Add confidence from a different perspective."""
        self.confidence_by_subject[perspective] = confidence
        self.change_history.append({
            "timestamp": time.time(),
            "action": "perspective_added",
            "perspective": perspective,
            "confidence": confidence
        })
    
    def set_nested_belief(self, belief_holder: str, belief_about: str):
        """Mark this as a nested belief (e.g., 'user believes that Anna believes...')."""
        self.nested_belief = True
        self.perspective = belief_holder
        self.belief_about = belief_about
        self.change_history.append({
            "timestamp": time.time(),
            "action": "nested_belief_set",
            "belief_holder": belief_holder,
            "belief_about": belief_about
        })
    
    def get_age_days(self) -> float:
        """Get the age of this fact in days."""
        if self.timestamp:
            return (time.time() - self.timestamp) / (24 * 3600)
        return 0.0
    
    def update_access(self):
        """Update access tracking."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def add_emotional_context(self, valence: float, arousal: float, emotion_tag: str = "", 
                            strength: float = 1.0, source: str = "system", mood_context: str = ""):
        """
        Add emotional context to this fact.
        
        Args:
            valence: Emotional valence (-1.0 to 1.0)
            arousal: Emotional arousal (0.0 to 1.0)
            emotion_tag: Emotion label (e.g., "curiosity", "frustration")
            strength: Strength of emotional association (0.0 to 1.0)
            source: Source of emotional tagging
            mood_context: Current mood context
        """
        # Clamp values to valid ranges
        self.emotion_valence = max(-1.0, min(1.0, valence))
        self.emotion_arousal = max(0.0, min(1.0, arousal))
        self.emotion_tag = emotion_tag
        self.emotional_strength = max(0.0, min(1.0, strength))
        self.emotion_source = source
        self.mood_context = mood_context
        
        # Record emotional tagging in change history
        self.change_history.append({
            "timestamp": time.time(),
            "action": "emotional_tagging",
            "valence": valence,
            "arousal": arousal,
            "emotion_tag": emotion_tag,
            "strength": strength,
            "source": source
        })
    
    def get_emotional_context(self) -> Dict[str, Any]:
        """
        Get the emotional context of this fact.
        
        Returns:
            Dictionary with emotional metadata
        """
        return {
            "valence": self.emotion_valence,
            "arousal": self.emotion_arousal,
            "emotion_tag": self.emotion_tag,
            "emotional_strength": self.emotional_strength,
            "emotion_source": self.emotion_source,
            "mood_context": self.mood_context,
            "has_emotion": self.emotion_valence is not None or self.emotion_arousal is not None
        }
    
    def clear_emotional_context(self):
        """Clear all emotional context from this fact."""
        self.emotion_valence = None
        self.emotion_arousal = None
        self.emotion_tag = None
        self.emotional_strength = 0.0
        self.emotion_source = None
        self.mood_context = None
        
        # Record clearing in change history
        self.change_history.append({
            "timestamp": time.time(),
            "action": "emotional_context_cleared"
        })
    
    def set_episode_id(self, episode_id: str, notes: str = ""):
        """
        Assign this fact to a life narrative episode.
        
        Args:
            episode_id: ID of the episode this fact belongs to
            notes: Optional notes about the assignment
        """
        self.episode_id = episode_id
        
        change_entry = {
            'action': 'set_episode_id',
            'timestamp': time.time(),
            'episode_id': episode_id,
            'notes': notes
        }
        self.change_history.append(change_entry)
    
    def get_episode_info(self) -> Dict[str, Any]:
        """Get episode assignment information."""
        return {
            'episode_id': self.episode_id,
            'has_episode': self.episode_id is not None
        }
    
    @classmethod
    def from_database_row(cls, row: tuple) -> 'EnhancedTripletFact':
        """
        Create EnhancedTripletFact from database row.
        
        Args:
            row: Database row tuple
            
        Returns:
            EnhancedTripletFact instance
        """
        # Assuming database row structure matches field order
        # This is a helper method for life narrative clustering
        try:
            fact = cls(
                subject=row[1] if len(row) > 1 else "unknown",
                predicate=row[2] if len(row) > 2 else "relates_to", 
                object=row[3] if len(row) > 3 else "unknown",
                timestamp=row[6] if len(row) > 6 else time.time(),
                confidence=row[5] if len(row) > 5 else 0.5,
                id=row[0] if len(row) > 0 else None,
                user_profile_id=row[7] if len(row) > 7 else None,
                session_id=row[8] if len(row) > 8 else None,
                
                # Optional fields that might not be in all database versions
                volatility_score=row[9] if len(row) > 9 else None,
                causal_strength=row[10] if len(row) > 10 else None,
                
                # Emotional fields (Phase 8)
                emotion_valence=row[11] if len(row) > 11 else None,
                emotion_arousal=row[12] if len(row) > 12 else None,
                emotion_tag=row[13] if len(row) > 13 else None,
                emotional_strength=row[14] if len(row) > 14 else 0.0,
                emotion_source=row[15] if len(row) > 15 else None,
                mood_context=row[16] if len(row) > 16 else None,
                
                # Episode field (Phase 10)
                episode_id=row[17] if len(row) > 17 else None
            )
            
            return fact
            
        except Exception as e:
            # Fallback for minimal rows
            fact = cls(
                subject=row[1] if len(row) > 1 else "unknown",
                predicate=row[2] if len(row) > 2 else "relates_to",
                object=row[3] if len(row) > 3 else "unknown",
                timestamp=row[6] if len(row) > 6 else time.time(),
                id=row[0] if len(row) > 0 else f"fact_{int(time.time() * 1000000)}"
            )
            return fact
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            # Core fields
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            
            # Contradiction and volatility
            "contradiction": self.contradiction,
            "contradicts_with": self.contradicts_with,
            "volatile": self.volatile,
            "volatility_score": self.volatility_score,
            
            # Session metadata
            "user_profile_id": self.user_profile_id,
            "session_id": self.session_id,
            "source_message_id": self.source_message_id,
            
            # Linguistic metadata
            "hedge_detected": self.hedge_detected,
            "intensifier_detected": self.intensifier_detected,
            "negation": self.negation,
            
            # Embedding
            "embedding": self.embedding,
            
            # PHASE 2: Causal & Temporal
            "cause": self.cause,
            "preceded_by": self.preceded_by,
            "causal_strength": self.causal_strength,
            
            # PHASE 2: Theory of Mind
            "perspective": self.perspective,
            "source": self.source,
            "confidence_by_subject": self.confidence_by_subject,
            "nested_belief": self.nested_belief,
            "belief_about": self.belief_about,
            
            # TOKEN-LEVEL ENTROPY & CAUSAL TRACKING
            "token_count": self.token_count,
            "token_entropy": self.token_entropy,
            "token_ids": self.token_ids,
            "token_hash": self.token_hash,
            "causal_token_propagation": self.causal_token_propagation,
            
            # Memory management
            "active": self.active,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            
            # History
            "change_history": self.change_history,
            
            # PHASE 8: Emotional Context
            "emotion_valence": self.emotion_valence,
            "emotion_arousal": self.emotion_arousal,
            "emotion_tag": self.emotion_tag,
            "emotional_strength": self.emotional_strength,
            "emotion_source": self.emotion_source,
            "mood_context": self.mood_context,
            
            # PHASE 10: Life Narrative
            "episode_id": self.episode_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedTripletFact':
        """Create from dictionary"""
        # Handle timestamp conversion (support both old datetime and new float formats)
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            try:
                # Try to parse as float first (new format)
                timestamp = float(timestamp)
            except ValueError:
                # Convert old datetime string to float
                dt = datetime.fromisoformat(timestamp)
                timestamp = dt.timestamp()
        elif timestamp is None:
            timestamp = time.time()
            
        return cls(
            # Core fields
            subject=data['subject'],
            predicate=data['predicate'],
            object=data['object'],
            timestamp=timestamp,
            confidence=data.get('confidence', 0.8),
            
            # Contradiction and volatility
            contradiction=data.get('contradiction', False),
            contradicts_with=data.get('contradicts_with', []),
            volatile=data.get('volatile', False),
            volatility_score=data.get('volatility_score', 0.0),
            
            # Session metadata
            id=data.get('id'),
            user_profile_id=data.get('user_profile_id'),
            session_id=data.get('session_id'),
            source_message_id=data.get('source_message_id'),
            
            # Linguistic metadata
            hedge_detected=data.get('hedge_detected', False),
            intensifier_detected=data.get('intensifier_detected', False),
            negation=data.get('negation', False),
            
            # Embedding
            embedding=data.get('embedding'),
            
            # PHASE 2: Causal & Temporal
            cause=data.get('cause'),
            preceded_by=data.get('preceded_by'),
            causal_strength=data.get('causal_strength', 0.0),
            
            # PHASE 2: Theory of Mind
            perspective=data.get('perspective', 'user'),
            source=data.get('source'),
            confidence_by_subject=data.get('confidence_by_subject', {}),
            nested_belief=data.get('nested_belief', False),
            belief_about=data.get('belief_about'),
            
            # TOKEN-LEVEL ENTROPY & CAUSAL TRACKING
            token_count=data.get('token_count'),
            token_entropy=data.get('token_entropy'),
            token_ids=data.get('token_ids'),
            token_hash=data.get('token_hash'),
            causal_token_propagation=data.get('causal_token_propagation', {}),
            
            # Memory management
            active=data.get('active', True),
            last_accessed=data.get('last_accessed'),
            access_count=data.get('access_count', 0),
            
            # History
            change_history=data.get('change_history', []),
            
            # PHASE 8: Emotional Context
            emotion_valence=data.get('emotion_valence'),
            emotion_arousal=data.get('emotion_arousal'),
            emotion_tag=data.get('emotion_tag'),
            emotional_strength=data.get('emotional_strength', 0.0),
            emotion_source=data.get('emotion_source'),
            mood_context=data.get('mood_context'),
            
            # PHASE 10: Life Narrative
            episode_id=data.get('episode_id')
        )
    
    def add_change(self, old_object: str, reason: str = "user_update"):
        """Track changes to this fact over time"""
        change_timestamp = time.time()
        self.change_history.append({
            "timestamp": change_timestamp,
            "old_object": old_object,
            "new_object": self.object,
            "reason": reason
        })
        
        # Update volatility based on change frequency
        if len(self.change_history) > 1:
            # Calculate time span of changes
            first_change_time = self.change_history[0]['timestamp']
            if isinstance(first_change_time, str):
                # Handle old datetime format
                first_change_time = datetime.fromisoformat(first_change_time).timestamp()
            
            time_span_days = (change_timestamp - first_change_time) / (24 * 3600)
            time_span_days = max(time_span_days, 1)  # Minimum 1 day
            
            # Volatility = changes per day
            self.volatility_score = len(self.change_history) / time_span_days
            self.volatile = self.volatility_score > 0.4  # Threshold from requirements
    
    def compute_token_entropy(self) -> float:
        """Compute entropy of token distribution."""
        if not self.token_ids:
            return 0.0
        
        counts = Counter(self.token_ids)
        total = len(self.token_ids)
        probs = [c / total for c in counts.values()]
        return -sum(p * math.log2(p) for p in probs)
    
    def generate_token_hash(self) -> str:
        """Generate hash of token sequence for quick comparison."""
        if not self.token_ids:
            return ""
        
        # Create a hash from the token sequence
        token_str = ",".join(map(str, self.token_ids))
        return str(hash(token_str))
    
    def set_tokens(self, tokens: List[int]):
        """Set token information and compute derived values."""
        self.token_ids = tokens
        self.token_count = len(tokens)
        self.token_entropy = self.compute_token_entropy()
        self.token_hash = self.generate_token_hash()
    
    def add_causal_token_propagation(self, target_fact_id: str, propagated_tokens: List[int]):
        """Track which tokens propagated to which facts (token→fact graph)."""
        if target_fact_id not in self.causal_token_propagation:
            self.causal_token_propagation[target_fact_id] = []
        self.causal_token_propagation[target_fact_id].extend(propagated_tokens)
    
    def get_token_jaccard_similarity(self, other_fact: 'EnhancedTripletFact') -> float:
        """Compute Jaccard similarity between token sets of two facts."""
        if not self.token_ids or not other_fact.token_ids:
            return 0.0
        
        set_a = set(self.token_ids)
        set_b = set(other_fact.token_ids)
        
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        
        return intersection / union if union > 0 else 0.0
    
    def get_token_overlap_ratio(self, other_fact: 'EnhancedTripletFact') -> float:
        """Compute token overlap ratio between two facts."""
        if not self.token_ids or not other_fact.token_ids:
            return 0.0
        
        set_a = set(self.token_ids)
        set_b = set(other_fact.token_ids)
        
        intersection = len(set_a.intersection(set_b))
        min_length = min(len(set_a), len(set_b))
        
        return intersection / min_length if min_length > 0 else 0.0


@dataclass 
class ContradictionRecord:
    """Record of a detected contradiction between facts"""
    id: Optional[int]
    fact_a_id: int
    fact_b_id: int
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_notes: Optional[str] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "fact_a_id": self.fact_a_id,
            "fact_b_id": self.fact_b_id,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "resolved": self.resolved,
            "resolution_notes": self.resolution_notes,
            "confidence": self.confidence
        } 


def compute_entropy(tokens: List[int]) -> float:
    """
    Compute entropy of token distribution.
    
    Args:
        tokens: List of token IDs
        
    Returns:
        Entropy value (higher = more diverse token distribution)
    """
    if not tokens:
        return 0.0
    
    counts = Counter(tokens)
    total = len(tokens)
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


def jaccard_similarity(tokens_a: List[int], tokens_b: List[int]) -> float:
    """
    Compute Jaccard similarity between two token lists.
    
    Args:
        tokens_a: First list of token IDs
        tokens_b: Second list of token IDs
        
    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    if not tokens_a or not tokens_b:
        return 0.0
    
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    
    return intersection / union if union > 0 else 0.0 