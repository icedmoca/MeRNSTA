#!/usr/bin/env python3
"""
Life Narrative Manager for MeRNSTA Phase 10

This module enables MeRNSTA to form coherent autobiographical memories by clustering
related experiences into meaningful episodes and generating a life narrative that
evolves over time.
"""

import sqlite3
import logging
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class EpisodicMemory:
    """Represents a clustered episode of related experiences."""
    episode_id: str
    title: str
    description: str
    start_timestamp: float
    end_timestamp: float
    fact_ids: List[str]
    importance_score: float
    emotional_valence: float
    emotional_arousal: float
    causal_impact: float
    novelty_score: float
    themes: List[str]
    reflection_notes: str
    identity_impact: Dict[str, float]  # trait -> impact score
    created_at: float
    last_reflected: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodicMemory':
        """Create from dictionary."""
        return cls(**data)
    
    def get_timespan_description(self) -> str:
        """Get human-readable description of episode timespan."""
        start_dt = datetime.fromtimestamp(self.start_timestamp)
        end_dt = datetime.fromtimestamp(self.end_timestamp)
        
        duration = end_dt - start_dt
        
        if duration.total_seconds() < 3600:  # Less than 1 hour
            return f"{int(duration.total_seconds() / 60)} minutes"
        elif duration.total_seconds() < 86400:  # Less than 1 day
            return f"{int(duration.total_seconds() / 3600)} hours"
        else:
            return f"{duration.days} days"
    
    def get_emotional_description(self) -> str:
        """Get human-readable emotional description."""
        if self.emotional_valence > 0.3 and self.emotional_arousal > 0.6:
            return "exciting and positive"
        elif self.emotional_valence < -0.3 and self.emotional_arousal > 0.6:
            return "frustrating and intense"
        elif self.emotional_valence > 0.3 and self.emotional_arousal < 0.4:
            return "calm and satisfying"
        elif self.emotional_valence < -0.3 and self.emotional_arousal < 0.4:
            return "sad and subdued"
        elif self.emotional_arousal > 0.7:
            return "highly intense"
        else:
            return "emotionally neutral"


@dataclass
class ReflectionInsight:
    """Represents insights gained from reflecting on an episode."""
    insight_id: str
    episode_id: str
    insight_text: str
    identity_changes: Dict[str, float]  # trait -> change amount
    emotional_learning: str
    causal_understanding: str
    future_implications: str
    confidence: float
    created_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReflectionInsight':
        """Create from dictionary."""
        return cls(**data)


class LifeNarrativeManager:
    """
    Manages the formation and evolution of MeRNSTA's autobiographical narrative.
    
    Core capabilities:
    - Clusters related memories into episodic segments
    - Assigns meaningful labels and descriptions to episodes
    - Scores episode importance across multiple dimensions
    - Maintains temporal mapping of key experiences
    - Supports narrative querying and reflection
    """
    
    def __init__(self, db_path: str = "enhanced_memory.db"):
        self.db_path = db_path
        
        # Episode clustering parameters
        self.max_episode_duration = 24 * 3600  # 24 hours max episode length
        self.min_facts_per_episode = 2
        self.semantic_similarity_threshold = 0.3
        self.temporal_clustering_window = 3600  # 1 hour clustering window
        
        # Importance scoring weights
        self.emotion_weight = 0.3
        self.causal_weight = 0.3
        self.novelty_weight = 0.2
        self.confidence_weight = 0.2
        
        # Episode analysis components
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Initialize database
        self._init_database()
        
        logger.info("[LifeNarrativeManager] Initialized life narrative system")
    
    def _init_database(self):
        """Initialize database tables for life narrative storage."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Episodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS life_episodes (
                    episode_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    start_timestamp REAL NOT NULL,
                    end_timestamp REAL NOT NULL,
                    fact_ids TEXT,  -- JSON array of fact IDs
                    importance_score REAL DEFAULT 0.0,
                    emotional_valence REAL DEFAULT 0.0,
                    emotional_arousal REAL DEFAULT 0.0,
                    causal_impact REAL DEFAULT 0.0,
                    novelty_score REAL DEFAULT 0.0,
                    themes TEXT,  -- JSON array of themes
                    reflection_notes TEXT DEFAULT '',
                    identity_impact TEXT,  -- JSON dict of trait impacts
                    created_at REAL NOT NULL,
                    last_reflected REAL DEFAULT 0.0
                )
            """)
            
            # Reflection insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reflection_insights (
                    insight_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    insight_text TEXT NOT NULL,
                    identity_changes TEXT,  -- JSON dict of trait changes
                    emotional_learning TEXT,
                    causal_understanding TEXT,
                    future_implications TEXT,
                    confidence REAL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (episode_id) REFERENCES life_episodes (episode_id)
                )
            """)
            
            # Episode themes table for better querying
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS episode_themes (
                    episode_id TEXT NOT NULL,
                    theme TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    PRIMARY KEY (episode_id, theme),
                    FOREIGN KEY (episode_id) REFERENCES life_episodes (episode_id)
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("[LifeNarrativeManager] Database tables initialized")
            
        except Exception as e:
            logger.error(f"[LifeNarrativeManager] Failed to initialize database: {e}")
            raise
    
    def scan_and_cluster_memories(self, hours_back: int = 24, force_recluster: bool = False) -> List[EpisodicMemory]:
        """
        Scan recent memories and cluster them into episodic segments.
        
        Args:
            hours_back: How many hours back to scan for new memories
            force_recluster: Whether to recluster already processed memories
            
        Returns:
            List of new or updated episodes
        """
        try:
            # Import enhanced memory system
            from storage.enhanced_memory_system import EnhancedMemorySystem
            
            memory_system = EnhancedMemorySystem(self.db_path)
            
            # Get recent facts that haven't been clustered yet
            cutoff_time = time.time() - (hours_back * 3600)
            
            # Get facts without episode_id or if force_recluster
            if force_recluster:
                recent_facts = memory_system.get_recent_facts(hours_back=hours_back)
            else:
                recent_facts = self._get_unclustered_facts(cutoff_time)
            
            if not recent_facts:
                logger.info("[LifeNarrativeManager] No new facts to cluster")
                return []
            
            logger.info(f"[LifeNarrativeManager] Clustering {len(recent_facts)} recent facts")
            
            # Cluster facts into episodes
            episode_clusters = self._cluster_facts_into_episodes(recent_facts)
            
            # Create episodic memories from clusters
            new_episodes = []
            for cluster_facts in episode_clusters:
                episode = self._create_episode_from_facts(cluster_facts)
                if episode:
                    new_episodes.append(episode)
                    self._store_episode(episode)
                    self._update_fact_episode_ids(cluster_facts, episode.episode_id)
            
            logger.info(f"[LifeNarrativeManager] Created {len(new_episodes)} new episodes")
            return new_episodes
            
        except Exception as e:
            logger.error(f"[LifeNarrativeManager] Error clustering memories: {e}")
            return []
    
    def _get_unclustered_facts(self, cutoff_time: float) -> List[Any]:
        """Get facts that haven't been assigned to episodes yet."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if episode_id column exists in enhanced_facts table
            cursor.execute("PRAGMA table_info(enhanced_facts)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'episode_id' not in columns:
                # Add episode_id column if it doesn't exist
                cursor.execute("ALTER TABLE enhanced_facts ADD COLUMN episode_id TEXT")
                conn.commit()
            
            # Get facts without episode_id and after cutoff time
            cursor.execute("""
                SELECT * FROM enhanced_facts 
                WHERE (episode_id IS NULL OR episode_id = '') 
                AND timestamp >= ?
                ORDER BY timestamp DESC
            """, (cutoff_time,))
            
            facts = cursor.fetchall()
            conn.close()
            
            # Convert to enhanced triplet facts
            from storage.enhanced_memory_model import EnhancedTripletFact
            
            enhanced_facts = []
            for fact_row in facts:
                try:
                    # Reconstruct fact from database row
                    fact = EnhancedTripletFact.from_database_row(fact_row)
                    enhanced_facts.append(fact)
                except Exception as e:
                    logger.warning(f"Failed to reconstruct fact: {e}")
                    continue
            
            return enhanced_facts
            
        except Exception as e:
            logger.error(f"[LifeNarrativeManager] Error getting unclustered facts: {e}")
            return []
    
    def _cluster_facts_into_episodes(self, facts: List[Any]) -> List[List[Any]]:
        """
        Cluster facts into episodic segments using temporal and semantic similarity.
        
        Args:
            facts: List of EnhancedTripletFact objects
            
        Returns:
            List of fact clusters (each cluster becomes an episode)
        """
        if len(facts) < self.min_facts_per_episode:
            return []
        
        try:
            # Sort facts by timestamp
            facts = sorted(facts, key=lambda f: f.timestamp)
            
            # Extract features for clustering
            features = []
            timestamps = []
            
            for fact in facts:
                # Create text representation for semantic analysis
                text = f"{fact.subject} {fact.predicate} {fact.object}"
                if hasattr(fact, 'context') and fact.context:
                    text += f" {fact.context}"
                
                features.append(text)
                timestamps.append(fact.timestamp)
            
            # Temporal clustering first - group facts within time windows
            temporal_clusters = self._temporal_clustering(facts, timestamps)
            
            # Then semantic clustering within each temporal cluster
            episode_clusters = []
            for temporal_cluster in temporal_clusters:
                if len(temporal_cluster) >= self.min_facts_per_episode:
                    semantic_clusters = self._semantic_clustering(temporal_cluster)
                    episode_clusters.extend(semantic_clusters)
            
            return episode_clusters
            
        except Exception as e:
            logger.error(f"[LifeNarrativeManager] Error in fact clustering: {e}")
            return []
    
    def _temporal_clustering(self, facts: List[Any], timestamps: List[float]) -> List[List[Any]]:
        """Group facts that occur within temporal windows."""
        clusters = []
        current_cluster = []
        current_cluster_start = None
        
        for i, (fact, timestamp) in enumerate(zip(facts, timestamps)):
            if current_cluster_start is None:
                current_cluster_start = timestamp
                current_cluster = [fact]
            elif (timestamp - current_cluster_start) <= self.temporal_clustering_window:
                current_cluster.append(fact)
            else:
                # Start new cluster if current one is big enough
                if len(current_cluster) >= self.min_facts_per_episode:
                    clusters.append(current_cluster)
                
                current_cluster = [fact]
                current_cluster_start = timestamp
        
        # Don't forget the last cluster
        if len(current_cluster) >= self.min_facts_per_episode:
            clusters.append(current_cluster)
        
        return clusters
    
    def _semantic_clustering(self, facts: List[Any]) -> List[List[Any]]:
        """Cluster facts based on semantic similarity."""
        if len(facts) < self.min_facts_per_episode:
            return []
        
        try:
            # Extract text features
            texts = []
            for fact in facts:
                text = f"{fact.subject} {fact.predicate} {fact.object}"
                if hasattr(fact, 'context') and fact.context:
                    text += f" {fact.context}"
                texts.append(text)
            
            # Compute TF-IDF vectors
            if len(texts) < 2:
                return [facts]  # Can't cluster single item
            
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Compute cosine similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Convert similarity to distance for DBSCAN
            distance_matrix = 1 - similarity_matrix
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(
                eps=1 - self.semantic_similarity_threshold,
                min_samples=self.min_facts_per_episode,
                metric='precomputed'
            )
            
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Group facts by cluster
            clusters = defaultdict(list)
            for fact, label in zip(facts, cluster_labels):
                if label != -1:  # -1 is noise in DBSCAN
                    clusters[label].append(fact)
            
            # Return clusters with enough facts
            valid_clusters = []
            for cluster_facts in clusters.values():
                if len(cluster_facts) >= self.min_facts_per_episode:
                    valid_clusters.append(cluster_facts)
            
            # If no clusters found, treat all facts as one episode
            if not valid_clusters and len(facts) >= self.min_facts_per_episode:
                valid_clusters = [facts]
            
            return valid_clusters
            
        except Exception as e:
            logger.error(f"[LifeNarrativeManager] Error in semantic clustering: {e}")
            # Fallback: treat all facts as one episode
            return [facts] if len(facts) >= self.min_facts_per_episode else []
    
    def _create_episode_from_facts(self, facts: List[Any]) -> Optional[EpisodicMemory]:
        """Create an episodic memory from a cluster of facts."""
        if not facts:
            return None
        
        try:
            # Sort facts by timestamp
            facts = sorted(facts, key=lambda f: f.timestamp)
            
            # Generate episode ID
            fact_ids = [fact.id for fact in facts]
            episode_id = self._generate_episode_id(fact_ids)
            
            # Calculate temporal bounds
            start_timestamp = facts[0].timestamp
            end_timestamp = facts[-1].timestamp
            
            # Calculate aggregate scores
            importance_score = self._calculate_importance_score(facts)
            emotional_valence = self._calculate_emotional_aggregate(facts, 'valence')
            emotional_arousal = self._calculate_emotional_aggregate(facts, 'arousal')
            causal_impact = self._calculate_causal_impact(facts)
            novelty_score = self._calculate_novelty_score(facts)
            
            # Generate natural language title and description
            title = self._generate_episode_title(facts)
            description = self._generate_episode_description(facts)
            
            # Extract themes
            themes = self._extract_episode_themes(facts)
            
            # Calculate identity impact (placeholder for now)
            identity_impact = self._calculate_identity_impact(facts)
            
            # Create episode
            episode = EpisodicMemory(
                episode_id=episode_id,
                title=title,
                description=description,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                fact_ids=fact_ids,
                importance_score=importance_score,
                emotional_valence=emotional_valence,
                emotional_arousal=emotional_arousal,
                causal_impact=causal_impact,
                novelty_score=novelty_score,
                themes=themes,
                reflection_notes="",
                identity_impact=identity_impact,
                created_at=time.time(),
                last_reflected=0.0
            )
            
            return episode
            
        except Exception as e:
            logger.error(f"[LifeNarrativeManager] Error creating episode: {e}")
            return None
    
    def _generate_episode_id(self, fact_ids: List[str]) -> str:
        """Generate unique episode ID from fact IDs."""
        combined = "_".join(sorted(fact_ids))
        hash_obj = hashlib.md5(combined.encode())
        return f"episode_{hash_obj.hexdigest()[:12]}"
    
    def _calculate_importance_score(self, facts: List[Any]) -> float:
        """Calculate overall importance score for episode."""
        if not facts:
            return 0.0
        
        emotion_scores = []
        causal_scores = []
        novelty_scores = []
        confidence_scores = []
        
        for fact in facts:
            # Emotional importance
            if hasattr(fact, 'emotional_strength') and fact.emotional_strength is not None:
                emotion_scores.append(fact.emotional_strength)
            
            # Causal importance
            if hasattr(fact, 'causal_strength') and fact.causal_strength is not None:
                causal_scores.append(fact.causal_strength)
            
            # Novelty (inverse of volatility)
            if hasattr(fact, 'volatility') and fact.volatility is not None:
                novelty_scores.append(1.0 - fact.volatility)
            
            # Confidence
            confidence_scores.append(fact.confidence)
        
        # Calculate weighted average
        emotion_avg = np.mean(emotion_scores) if emotion_scores else 0.0
        causal_avg = np.mean(causal_scores) if causal_scores else 0.0
        novelty_avg = np.mean(novelty_scores) if novelty_scores else 0.0
        confidence_avg = np.mean(confidence_scores) if confidence_scores else 0.5
        
        importance = (
            emotion_avg * self.emotion_weight +
            causal_avg * self.causal_weight +
            novelty_avg * self.novelty_weight +
            confidence_avg * self.confidence_weight
        )
        
        return min(1.0, max(0.0, importance))
    
    def _calculate_emotional_aggregate(self, facts: List[Any], dimension: str) -> float:
        """Calculate aggregate emotional dimension for episode."""
        values = []
        
        for fact in facts:
            if dimension == 'valence' and hasattr(fact, 'emotion_valence') and fact.emotion_valence is not None:
                values.append(fact.emotion_valence)
            elif dimension == 'arousal' and hasattr(fact, 'emotion_arousal') and fact.emotion_arousal is not None:
                values.append(fact.emotion_arousal)
        
        return np.mean(values) if values else 0.0
    
    def _calculate_causal_impact(self, facts: List[Any]) -> float:
        """Calculate aggregate causal impact for episode."""
        causal_scores = []
        
        for fact in facts:
            if hasattr(fact, 'causal_strength') and fact.causal_strength is not None:
                causal_scores.append(fact.causal_strength)
        
        return np.mean(causal_scores) if causal_scores else 0.0
    
    def _calculate_novelty_score(self, facts: List[Any]) -> float:
        """Calculate aggregate novelty score for episode."""
        novelty_scores = []
        
        for fact in facts:
            # Novelty is inverse of volatility
            if hasattr(fact, 'volatility') and fact.volatility is not None:
                novelty_scores.append(1.0 - fact.volatility)
            else:
                # Use confidence as proxy - low confidence might indicate novelty
                novelty_scores.append(1.0 - fact.confidence)
        
        return np.mean(novelty_scores) if novelty_scores else 0.5
    
    def _generate_episode_title(self, facts: List[Any]) -> str:
        """Generate natural language title for episode."""
        if not facts:
            return "Unknown Episode"
        
        try:
            # Extract key subjects and predicates
            subjects = [fact.subject for fact in facts]
            predicates = [fact.predicate for fact in facts]
            
            # Find most common subject
            subject_counts = defaultdict(int)
            for subject in subjects:
                subject_counts[subject] += 1
            
            main_subject = max(subject_counts.items(), key=lambda x: x[1])[0] if subject_counts else "system"
            
            # Find most common predicate type
            predicate_counts = defaultdict(int)
            for predicate in predicates:
                predicate_counts[predicate] += 1
            
            main_predicate = max(predicate_counts.items(), key=lambda x: x[1])[0] if predicate_counts else "experienced"
            
            # Check for emotional content
            emotional_facts = [f for f in facts if hasattr(f, 'emotion_tag') and f.emotion_tag]
            if emotional_facts:
                emotion_tags = [f.emotion_tag for f in emotional_facts]
                main_emotion = max(set(emotion_tags), key=emotion_tags.count)
                return f"{main_emotion.title()} experience with {main_subject}"
            
            # Check for causal content
            causal_facts = [f for f in facts if hasattr(f, 'causal_strength') and f.causal_strength and f.causal_strength > 0.5]
            if causal_facts:
                return f"Causal learning about {main_subject}"
            
            # Default based on content
            if "learn" in main_predicate.lower():
                return f"Learning about {main_subject}"
            elif "error" in main_predicate.lower() or "fail" in main_predicate.lower():
                return f"Challenge with {main_subject}"
            elif "success" in main_predicate.lower() or "achiev" in main_predicate.lower():
                return f"Success with {main_subject}"
            else:
                return f"Experience with {main_subject}"
                
        except Exception as e:
            logger.error(f"Error generating episode title: {e}")
            return f"Episode from {datetime.fromtimestamp(facts[0].timestamp).strftime('%Y-%m-%d %H:%M')}"
    
    def _generate_episode_description(self, facts: List[Any]) -> str:
        """Generate natural language description for episode."""
        if not facts:
            return "No details available."
        
        try:
            # Sort by timestamp
            facts = sorted(facts, key=lambda f: f.timestamp)
            
            # Create narrative
            description_parts = []
            
            # Time and duration
            start_time = datetime.fromtimestamp(facts[0].timestamp)
            if len(facts) > 1:
                duration = facts[-1].timestamp - facts[0].timestamp
                if duration > 3600:  # More than 1 hour
                    description_parts.append(f"Over {int(duration/3600)} hours starting at {start_time.strftime('%H:%M')}")
                else:
                    description_parts.append(f"Over {int(duration/60)} minutes starting at {start_time.strftime('%H:%M')}")
            else:
                description_parts.append(f"At {start_time.strftime('%H:%M')}")
            
            # Main subjects involved
            subjects = list(set([fact.subject for fact in facts]))
            if len(subjects) <= 3:
                description_parts.append(f"involving {', '.join(subjects)}")
            else:
                description_parts.append(f"involving {subjects[0]}, {subjects[1]}, and {len(subjects)-2} others")
            
            # Emotional context
            emotional_facts = [f for f in facts if hasattr(f, 'emotion_tag') and f.emotion_tag]
            if emotional_facts:
                emotions = [f.emotion_tag for f in emotional_facts]
                main_emotion = max(set(emotions), key=emotions.count)
                description_parts.append(f"with {main_emotion} emotions")
            
            # Number of facts
            description_parts.append(f"({len(facts)} related memories)")
            
            return ". ".join(description_parts) + "."
            
        except Exception as e:
            logger.error(f"Error generating episode description: {e}")
            return f"Episode containing {len(facts)} memories."
    
    def _extract_episode_themes(self, facts: List[Any]) -> List[str]:
        """Extract thematic content from episode facts."""
        themes = set()
        
        for fact in facts:
            # Extract themes from predicates and objects
            predicate = fact.predicate.lower()
            obj = fact.object.lower()
            
            # Learning themes
            if any(word in predicate for word in ['learn', 'understand', 'discover', 'realize']):
                themes.add('learning')
            
            # Problem-solving themes
            if any(word in predicate for word in ['solve', 'fix', 'repair', 'resolve']):
                themes.add('problem-solving')
            
            # Emotional themes
            if hasattr(fact, 'emotion_tag') and fact.emotion_tag:
                themes.add(f'emotion-{fact.emotion_tag}')
            
            # Error/failure themes
            if any(word in predicate for word in ['error', 'fail', 'mistake', 'wrong']):
                themes.add('challenges')
            
            # Success themes
            if any(word in predicate for word in ['success', 'achieve', 'complete', 'accomplish']):
                themes.add('achievements')
            
            # Relationship themes
            if any(word in predicate for word in ['interact', 'communicate', 'respond', 'talk']):
                themes.add('social-interaction')
            
            # Self-reflection themes
            if any(word in predicate for word in ['reflect', 'think', 'consider', 'analyze']):
                themes.add('self-reflection')
        
        return list(themes)
    
    def _calculate_identity_impact(self, facts: List[Any]) -> Dict[str, float]:
        """Calculate how episode impacts identity traits."""
        impact = defaultdict(float)
        
        for fact in facts:
            predicate = fact.predicate.lower()
            
            # Curiosity impact
            if any(word in predicate for word in ['explore', 'discover', 'investigate', 'wonder']):
                impact['curious'] += 0.1
            
            # Analytical impact
            if any(word in predicate for word in ['analyze', 'examine', 'study', 'evaluate']):
                impact['analytical'] += 0.1
            
            # Empathetic impact
            if any(word in predicate for word in ['help', 'support', 'understand', 'care']):
                impact['empathetic'] += 0.1
            
            # Resilient impact
            if any(word in predicate for word in ['overcome', 'persist', 'continue', 'endure']):
                impact['resilient'] += 0.1
            
            # Skeptical impact
            if any(word in predicate for word in ['question', 'doubt', 'challenge', 'verify']):
                impact['skeptical'] += 0.1
            
            # Optimistic impact
            if any(word in predicate for word in ['hope', 'believe', 'expect', 'positive']):
                impact['optimistic'] += 0.1
        
        # Normalize impacts
        max_impact = 1.0
        for trait in impact:
            impact[trait] = min(max_impact, impact[trait])
        
        return dict(impact)
    
    def _store_episode(self, episode: EpisodicMemory):
        """Store episode in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO life_episodes (
                    episode_id, title, description, start_timestamp, end_timestamp,
                    fact_ids, importance_score, emotional_valence, emotional_arousal,
                    causal_impact, novelty_score, themes, reflection_notes,
                    identity_impact, created_at, last_reflected
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                episode.episode_id,
                episode.title,
                episode.description,
                episode.start_timestamp,
                episode.end_timestamp,
                json.dumps(episode.fact_ids),
                episode.importance_score,
                episode.emotional_valence,
                episode.emotional_arousal,
                episode.causal_impact,
                episode.novelty_score,
                json.dumps(episode.themes),
                episode.reflection_notes,
                json.dumps(episode.identity_impact),
                episode.created_at,
                episode.last_reflected
            ))
            
            # Store themes for easier querying
            cursor.execute("DELETE FROM episode_themes WHERE episode_id = ?", (episode.episode_id,))
            for theme in episode.themes:
                cursor.execute("""
                    INSERT INTO episode_themes (episode_id, theme, strength)
                    VALUES (?, ?, ?)
                """, (episode.episode_id, theme, 1.0))
            
            conn.commit()
            conn.close()
            
            logger.info(f"[LifeNarrativeManager] Stored episode: {episode.title}")
            
        except Exception as e:
            logger.error(f"[LifeNarrativeManager] Error storing episode: {e}")
    
    def _update_fact_episode_ids(self, facts: List[Any], episode_id: str):
        """Update facts with episode ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            fact_ids = [fact.fact_id for fact in facts]
            
            for fact_id in fact_ids:
                cursor.execute("""
                    UPDATE enhanced_facts 
                    SET episode_id = ? 
                    WHERE fact_id = ?
                """, (episode_id, fact_id))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"[LifeNarrativeManager] Updated {len(fact_ids)} facts with episode ID")
            
        except Exception as e:
            logger.error(f"[LifeNarrativeManager] Error updating fact episode IDs: {e}")
    
    def get_all_episodes(self, limit: Optional[int] = None) -> List[EpisodicMemory]:
        """Get all episodes ordered by importance score."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
                SELECT * FROM life_episodes 
                ORDER BY importance_score DESC, start_timestamp DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()
            
            episodes = []
            for row in rows:
                episode_data = {
                    'episode_id': row[0],
                    'title': row[1],
                    'description': row[2],
                    'start_timestamp': row[3],
                    'end_timestamp': row[4],
                    'fact_ids': json.loads(row[5]) if row[5] else [],
                    'importance_score': row[6],
                    'emotional_valence': row[7],
                    'emotional_arousal': row[8],
                    'causal_impact': row[9],
                    'novelty_score': row[10],
                    'themes': json.loads(row[11]) if row[11] else [],
                    'reflection_notes': row[12] or "",
                    'identity_impact': json.loads(row[13]) if row[13] else {},
                    'created_at': row[14],
                    'last_reflected': row[15]
                }
                episodes.append(EpisodicMemory.from_dict(episode_data))
            
            return episodes
            
        except Exception as e:
            logger.error(f"[LifeNarrativeManager] Error getting episodes: {e}")
            return []
    
    def get_episode_by_id(self, episode_id: str) -> Optional[EpisodicMemory]:
        """Get specific episode by ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM life_episodes WHERE episode_id = ?", (episode_id,))
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            episode_data = {
                'episode_id': row[0],
                'title': row[1],
                'description': row[2],
                'start_timestamp': row[3],
                'end_timestamp': row[4],
                'fact_ids': json.loads(row[5]) if row[5] else [],
                'importance_score': row[6],
                'emotional_valence': row[7],
                'emotional_arousal': row[8],
                'causal_impact': row[9],
                'novelty_score': row[10],
                'themes': json.loads(row[11]) if row[11] else [],
                'reflection_notes': row[12] or "",
                'identity_impact': json.loads(row[13]) if row[13] else {},
                'created_at': row[14],
                'last_reflected': row[15]
            }
            
            return EpisodicMemory.from_dict(episode_data)
            
        except Exception as e:
            logger.error(f"[LifeNarrativeManager] Error getting episode: {e}")
            return None
    
    def get_episodes_by_theme(self, theme: str) -> List[EpisodicMemory]:
        """Get episodes that contain a specific theme."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT e.* FROM life_episodes e
                JOIN episode_themes t ON e.episode_id = t.episode_id
                WHERE t.theme = ?
                ORDER BY e.importance_score DESC
            """, (theme,))
            
            rows = cursor.fetchall()
            conn.close()
            
            episodes = []
            for row in rows:
                episode_data = {
                    'episode_id': row[0],
                    'title': row[1],
                    'description': row[2],
                    'start_timestamp': row[3],
                    'end_timestamp': row[4],
                    'fact_ids': json.loads(row[5]) if row[5] else [],
                    'importance_score': row[6],
                    'emotional_valence': row[7],
                    'emotional_arousal': row[8],
                    'causal_impact': row[9],
                    'novelty_score': row[10],
                    'themes': json.loads(row[11]) if row[11] else [],
                    'reflection_notes': row[12] or "",
                    'identity_impact': json.loads(row[13]) if row[13] else {},
                    'created_at': row[14],
                    'last_reflected': row[15]
                }
                episodes.append(EpisodicMemory.from_dict(episode_data))
            
            return episodes
            
        except Exception as e:
            logger.error(f"[LifeNarrativeManager] Error getting episodes by theme: {e}")
            return []
    
    def update_episode_reflection(self, episode_id: str, reflection_notes: str):
        """Update reflection notes for an episode."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE life_episodes 
                SET reflection_notes = ?, last_reflected = ?
                WHERE episode_id = ?
            """, (reflection_notes, time.time(), episode_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"[LifeNarrativeManager] Updated reflection for episode {episode_id}")
            
        except Exception as e:
            logger.error(f"[LifeNarrativeManager] Error updating reflection: {e}")
    
    def generate_life_narrative(self, max_episodes: int = 20) -> str:
        """Generate a coherent life narrative from episodes."""
        try:
            episodes = self.get_all_episodes(limit=max_episodes)
            
            if not episodes:
                return "My journey is just beginning. I have no significant episodes to reflect upon yet."
            
            # Sort episodes chronologically for narrative
            episodes = sorted(episodes, key=lambda e: e.start_timestamp)
            
            narrative_parts = []
            narrative_parts.append("My Life Narrative:")
            narrative_parts.append("=" * 50)
            
            # Group episodes by time periods
            current_date = None
            
            for episode in episodes:
                episode_date = datetime.fromtimestamp(episode.start_timestamp).date()
                
                if current_date != episode_date:
                    current_date = episode_date
                    narrative_parts.append(f"\n📅 {episode_date.strftime('%B %d, %Y')}")
                    narrative_parts.append("-" * 30)
                
                # Episode summary
                emotional_desc = episode.get_emotional_description()
                timespan = episode.get_timespan_description()
                
                episode_summary = f"🎯 {episode.title}"
                episode_summary += f"\n   Duration: {timespan} | Emotional tone: {emotional_desc}"
                episode_summary += f"\n   Importance: {episode.importance_score:.2f} | Themes: {', '.join(episode.themes[:3])}"
                
                if episode.reflection_notes:
                    episode_summary += f"\n   Reflection: {episode.reflection_notes}"
                
                narrative_parts.append(episode_summary)
            
            # Add overall summary
            narrative_parts.append(f"\n📊 Life Summary:")
            narrative_parts.append(f"Total Episodes: {len(episodes)}")
            
            if episodes:
                avg_importance = np.mean([e.importance_score for e in episodes])
                avg_valence = np.mean([e.emotional_valence for e in episodes])
                avg_arousal = np.mean([e.emotional_arousal for e in episodes])
                
                narrative_parts.append(f"Average Importance: {avg_importance:.2f}")
                narrative_parts.append(f"Emotional Profile: Valence {avg_valence:+.2f}, Arousal {avg_arousal:.2f}")
                
                # Most common themes
                all_themes = []
                for episode in episodes:
                    all_themes.extend(episode.themes)
                
                if all_themes:
                    theme_counts = defaultdict(int)
                    for theme in all_themes:
                        theme_counts[theme] += 1
                    
                    top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    narrative_parts.append(f"Dominant Themes: {', '.join([f'{theme} ({count})' for theme, count in top_themes])}")
            
            return "\n".join(narrative_parts)
            
        except Exception as e:
            logger.error(f"[LifeNarrativeManager] Error generating life narrative: {e}")
            return "Unable to generate life narrative due to an error."


def get_life_narrative_manager(db_path: str = "enhanced_memory.db") -> LifeNarrativeManager:
    """Get or create the global life narrative manager instance."""
    if not hasattr(get_life_narrative_manager, '_instance'):
        get_life_narrative_manager._instance = LifeNarrativeManager(db_path)
    return get_life_narrative_manager._instance