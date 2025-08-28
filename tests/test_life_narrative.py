#!/usr/bin/env python3
"""
Test suite for Phase 10 Life Narrative & Reflective Memory
"""

import unittest
import tempfile
import os
import time
import json
from unittest.mock import patch, MagicMock

from storage.life_narrative import LifeNarrativeManager, EpisodicMemory, ReflectionInsight, get_life_narrative_manager
from agents.reflective_engine import ReflectiveReplayEngine, ReflectionSession, get_reflective_engine
from storage.enhanced_memory_model import EnhancedTripletFact


class TestEpisodicMemory(unittest.TestCase):
    """Test cases for EpisodicMemory dataclass."""
    
    def test_episodic_memory_creation(self):
        """Test creating EpisodicMemory with valid values."""
        current_time = time.time()
        
        episode = EpisodicMemory(
            episode_id="test_episode_123",
            title="Test Episode",
            description="A test episode for validation",
            start_timestamp=current_time,
            end_timestamp=current_time + 3600,
            fact_ids=["fact1", "fact2", "fact3"],
            importance_score=0.8,
            emotional_valence=0.5,
            emotional_arousal=0.6,
            causal_impact=0.7,
            novelty_score=0.4,
            themes=["learning", "testing"],
            reflection_notes="Good test episode",
            identity_impact={"curious": 0.2, "analytical": 0.1},
            created_at=current_time,
            last_reflected=0.0
        )
        
        self.assertEqual(episode.episode_id, "test_episode_123")
        self.assertEqual(episode.title, "Test Episode")
        self.assertEqual(len(episode.fact_ids), 3)
        self.assertEqual(episode.importance_score, 0.8)
        self.assertIn("learning", episode.themes)
    
    def test_episodic_memory_serialization(self):
        """Test EpisodicMemory to_dict and from_dict."""
        current_time = time.time()
        
        episode = EpisodicMemory(
            episode_id="test_123",
            title="Serialization Test",
            description="Testing serialization",
            start_timestamp=current_time,
            end_timestamp=current_time + 1800,
            fact_ids=["fact1", "fact2"],
            importance_score=0.6,
            emotional_valence=0.2,
            emotional_arousal=0.4,
            causal_impact=0.3,
            novelty_score=0.5,
            themes=["testing"],
            reflection_notes="",
            identity_impact={"curious": 0.1},
            created_at=current_time,
            last_reflected=current_time
        )
        
        # Test to_dict
        episode_dict = episode.to_dict()
        self.assertIn('episode_id', episode_dict)
        self.assertIn('title', episode_dict)
        self.assertIn('themes', episode_dict)
        self.assertEqual(episode_dict['episode_id'], "test_123")
        
        # Test from_dict
        reconstructed = EpisodicMemory.from_dict(episode_dict)
        self.assertEqual(reconstructed.episode_id, episode.episode_id)
        self.assertEqual(reconstructed.title, episode.title)
        self.assertEqual(reconstructed.themes, episode.themes)
        self.assertEqual(reconstructed.importance_score, episode.importance_score)
    
    def test_episodic_memory_descriptions(self):
        """Test EpisodicMemory description methods."""
        current_time = time.time()
        
        # Test short duration
        episode_short = EpisodicMemory(
            episode_id="short",
            title="Short Episode",
            description="Short test",
            start_timestamp=current_time,
            end_timestamp=current_time + 1800,  # 30 minutes
            fact_ids=["fact1"],
            importance_score=0.5,
            emotional_valence=0.3,
            emotional_arousal=0.7,
            causal_impact=0.2,
            novelty_score=0.3,
            themes=["test"],
            reflection_notes="",
            identity_impact={},
            created_at=current_time,
            last_reflected=0.0
        )
        
        timespan = episode_short.get_timespan_description()
        self.assertIn("minutes", timespan)
        
        emotional = episode_short.get_emotional_description()
        # High arousal with neutral valence should reflect high energy
        self.assertTrue(any(word in emotional.lower() for word in ["intense", "highly", "energetic", "neutral"]))
        
        # Test positive valence episode
        episode_positive = EpisodicMemory(
            episode_id="positive",
            title="Positive Episode",
            description="Positive test",
            start_timestamp=current_time,
            end_timestamp=current_time + 7200,  # 2 hours
            fact_ids=["fact1"],
            importance_score=0.5,
            emotional_valence=0.6,
            emotional_arousal=0.8,
            causal_impact=0.2,
            novelty_score=0.3,
            themes=["test"],
            reflection_notes="",
            identity_impact={},
            created_at=current_time,
            last_reflected=0.0
        )
        
        timespan = episode_positive.get_timespan_description()
        self.assertIn("hours", timespan)
        
        emotional = episode_positive.get_emotional_description()
        self.assertIn("exciting", emotional.lower())  # High positive valence + arousal


class TestLifeNarrativeManager(unittest.TestCase):
    """Test cases for LifeNarrativeManager."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.test_db.close()
        self.db_path = self.test_db.name
        
        self.narrative_manager = LifeNarrativeManager(self.db_path)
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_narrative_manager_initialization(self):
        """Test LifeNarrativeManager initialization."""
        self.assertIsNotNone(self.narrative_manager.vectorizer)
        self.assertGreater(self.narrative_manager.max_episode_duration, 0)
        self.assertGreater(self.narrative_manager.min_facts_per_episode, 0)
        
        # Test database table creation
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        self.assertIn('life_episodes', tables)
        self.assertIn('reflection_insights', tables)
        self.assertIn('episode_themes', tables)
        
        conn.close()
    
    def test_episode_id_generation(self):
        """Test episode ID generation."""
        fact_ids = ["fact1", "fact2", "fact3"]
        episode_id = self.narrative_manager._generate_episode_id(fact_ids)
        
        self.assertIsInstance(episode_id, str)
        self.assertTrue(episode_id.startswith('episode_'))
        self.assertEqual(len(episode_id), 20)  # 'episode_' + 12 char hash
        
        # Same input should generate same ID
        episode_id2 = self.narrative_manager._generate_episode_id(fact_ids)
        self.assertEqual(episode_id, episode_id2)
        
        # Different input should generate different ID
        different_fact_ids = ["fact4", "fact5", "fact6"]
        episode_id3 = self.narrative_manager._generate_episode_id(different_fact_ids)
        self.assertNotEqual(episode_id, episode_id3)
    
    def test_importance_score_calculation(self):
        """Test importance score calculation."""
        # Create mock facts with different characteristics
        current_time = time.time()
        
        # High importance fact
        high_fact = EnhancedTripletFact(
            subject="system",
            predicate="achieved",
            object="breakthrough",
            timestamp=current_time,
            confidence=0.9,
            emotional_strength=0.8,
            causal_strength=0.7,
            volatility_score=0.2  # Low volatility = high novelty
        )
        
        # Low importance fact  
        low_fact = EnhancedTripletFact(
            subject="system",
            predicate="performed",
            object="routine_task",
            timestamp=current_time,
            confidence=0.5,
            emotional_strength=0.1,
            causal_strength=0.1,
            volatility_score=0.8  # High volatility = low novelty
        )
        
        high_score = self.narrative_manager._calculate_importance_score([high_fact])
        low_score = self.narrative_manager._calculate_importance_score([low_fact])
        
        self.assertGreater(high_score, low_score)
        self.assertLessEqual(high_score, 1.0)
        self.assertGreaterEqual(low_score, 0.0)
    
    def test_episode_title_generation(self):
        """Test episode title generation."""
        current_time = time.time()
        
        # Learning episode
        learning_fact = EnhancedTripletFact(
            subject="system",
            predicate="learned",
            object="new_concept",
            timestamp=current_time,
            emotion_tag="curiosity"
        )
        
        title = self.narrative_manager._generate_episode_title([learning_fact])
        self.assertIn("system", title.lower())
        
        # Emotional episode
        emotional_fact = EnhancedTripletFact(
            subject="user",
            predicate="expressed",
            object="frustration",
            timestamp=current_time,
            emotion_tag="frustration"
        )
        
        title = self.narrative_manager._generate_episode_title([emotional_fact])
        self.assertIn("frustration", title.lower())
    
    def test_theme_extraction(self):
        """Test episode theme extraction."""
        current_time = time.time()
        
        facts = [
            EnhancedTripletFact(
                subject="system",
                predicate="learned",
                object="concept",
                timestamp=current_time,
                emotion_tag="curiosity"
            ),
            EnhancedTripletFact(
                subject="system",
                predicate="solved",
                object="problem",
                timestamp=current_time
            ),
            EnhancedTripletFact(
                subject="system",
                predicate="failed",
                object="attempt",
                timestamp=current_time,
                emotion_tag="frustration"
            )
        ]
        
        themes = self.narrative_manager._extract_episode_themes(facts)
        
        self.assertIn("learning", themes)
        self.assertIn("problem-solving", themes)
        self.assertIn("challenges", themes)
        self.assertIn("emotion-curiosity", themes)
        self.assertIn("emotion-frustration", themes)
    
    def test_store_and_retrieve_episode(self):
        """Test storing and retrieving episodes."""
        current_time = time.time()
        
        episode = EpisodicMemory(
            episode_id="test_store_123",
            title="Store Test Episode",
            description="Testing storage functionality",
            start_timestamp=current_time,
            end_timestamp=current_time + 3600,
            fact_ids=["fact1", "fact2"],
            importance_score=0.7,
            emotional_valence=0.4,
            emotional_arousal=0.5,
            causal_impact=0.6,
            novelty_score=0.3,
            themes=["testing", "storage"],
            reflection_notes="Test reflection",
            identity_impact={"analytical": 0.2},
            created_at=current_time,
            last_reflected=current_time
        )
        
        # Store episode
        self.narrative_manager._store_episode(episode)
        
        # Retrieve episode
        retrieved = self.narrative_manager.get_episode_by_id("test_store_123")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.episode_id, episode.episode_id)
        self.assertEqual(retrieved.title, episode.title)
        self.assertEqual(retrieved.themes, episode.themes)
        self.assertEqual(retrieved.importance_score, episode.importance_score)
    
    def test_get_all_episodes(self):
        """Test getting all episodes with ordering."""
        current_time = time.time()
        
        # Create episodes with different importance scores
        episode1 = EpisodicMemory(
            episode_id="ep1",
            title="Low Importance",
            description="Low importance episode",
            start_timestamp=current_time,
            end_timestamp=current_time + 1800,
            fact_ids=["fact1"],
            importance_score=0.3,
            emotional_valence=0.0,
            emotional_arousal=0.0,
            causal_impact=0.0,
            novelty_score=0.0,
            themes=["test"],
            reflection_notes="",
            identity_impact={},
            created_at=current_time,
            last_reflected=0.0
        )
        
        episode2 = EpisodicMemory(
            episode_id="ep2",
            title="High Importance",
            description="High importance episode",
            start_timestamp=current_time + 3600,
            end_timestamp=current_time + 5400,
            fact_ids=["fact2"],
            importance_score=0.9,
            emotional_valence=0.0,
            emotional_arousal=0.0,
            causal_impact=0.0,
            novelty_score=0.0,
            themes=["test"],
            reflection_notes="",
            identity_impact={},
            created_at=current_time + 3600,
            last_reflected=0.0
        )
        
        # Store episodes
        self.narrative_manager._store_episode(episode1)
        self.narrative_manager._store_episode(episode2)
        
        # Retrieve all episodes
        all_episodes = self.narrative_manager.get_all_episodes()
        
        self.assertEqual(len(all_episodes), 2)
        # Should be ordered by importance (high to low)
        self.assertEqual(all_episodes[0].episode_id, "ep2")
        self.assertEqual(all_episodes[1].episode_id, "ep1")
        
        # Test limit
        limited_episodes = self.narrative_manager.get_all_episodes(limit=1)
        self.assertEqual(len(limited_episodes), 1)
        self.assertEqual(limited_episodes[0].episode_id, "ep2")
    
    def test_get_episodes_by_theme(self):
        """Test getting episodes by theme."""
        current_time = time.time()
        
        episode1 = EpisodicMemory(
            episode_id="theme_ep1",
            title="Learning Episode",
            description="Learning focused episode",
            start_timestamp=current_time,
            end_timestamp=current_time + 1800,
            fact_ids=["fact1"],
            importance_score=0.6,
            emotional_valence=0.0,
            emotional_arousal=0.0,
            causal_impact=0.0,
            novelty_score=0.0,
            themes=["learning", "curiosity"],
            reflection_notes="",
            identity_impact={},
            created_at=current_time,
            last_reflected=0.0
        )
        
        episode2 = EpisodicMemory(
            episode_id="theme_ep2",
            title="Problem Solving Episode",
            description="Problem solving focused episode",
            start_timestamp=current_time + 3600,
            end_timestamp=current_time + 5400,
            fact_ids=["fact2"],
            importance_score=0.7,
            emotional_valence=0.0,
            emotional_arousal=0.0,
            causal_impact=0.0,
            novelty_score=0.0,
            themes=["problem-solving", "challenges"],
            reflection_notes="",
            identity_impact={},
            created_at=current_time + 3600,
            last_reflected=0.0
        )
        
        # Store episodes
        self.narrative_manager._store_episode(episode1)
        self.narrative_manager._store_episode(episode2)
        
        # Get episodes by theme
        learning_episodes = self.narrative_manager.get_episodes_by_theme("learning")
        problem_episodes = self.narrative_manager.get_episodes_by_theme("problem-solving")
        
        self.assertEqual(len(learning_episodes), 1)
        self.assertEqual(learning_episodes[0].episode_id, "theme_ep1")
        
        self.assertEqual(len(problem_episodes), 1)
        self.assertEqual(problem_episodes[0].episode_id, "theme_ep2")
        
        # Non-existent theme
        none_episodes = self.narrative_manager.get_episodes_by_theme("nonexistent")
        self.assertEqual(len(none_episodes), 0)
    
    def test_update_episode_reflection(self):
        """Test updating episode reflection notes."""
        current_time = time.time()
        
        episode = EpisodicMemory(
            episode_id="reflection_test",
            title="Reflection Test",
            description="Testing reflection updates",
            start_timestamp=current_time,
            end_timestamp=current_time + 1800,
            fact_ids=["fact1"],
            importance_score=0.5,
            emotional_valence=0.0,
            emotional_arousal=0.0,
            causal_impact=0.0,
            novelty_score=0.0,
            themes=["test"],
            reflection_notes="",
            identity_impact={},
            created_at=current_time,
            last_reflected=0.0
        )
        
        # Store episode
        self.narrative_manager._store_episode(episode)
        
        # Update reflection
        reflection_text = "This was a valuable learning experience."
        self.narrative_manager.update_episode_reflection("reflection_test", reflection_text)
        
        # Retrieve and verify
        updated_episode = self.narrative_manager.get_episode_by_id("reflection_test")
        self.assertEqual(updated_episode.reflection_notes, reflection_text)
        self.assertGreater(updated_episode.last_reflected, 0)
    
    def test_generate_life_narrative(self):
        """Test generating complete life narrative."""
        current_time = time.time()
        
        # Create a few episodes
        episodes = []
        for i in range(3):
            episode = EpisodicMemory(
                episode_id=f"narrative_ep{i}",
                title=f"Episode {i+1}",
                description=f"Test episode {i+1}",
                start_timestamp=current_time + (i * 3600),
                end_timestamp=current_time + ((i+1) * 3600),
                fact_ids=[f"fact{i}"],
                importance_score=0.5 + (i * 0.1),
                emotional_valence=0.2 * i - 0.2,
                emotional_arousal=0.3 + (i * 0.1),
                causal_impact=0.4,
                novelty_score=0.3,
                themes=[f"theme{i}", "common"],
                reflection_notes=f"Reflection {i}",
                identity_impact={"curious": 0.1 * i},
                created_at=current_time + (i * 3600),
                last_reflected=current_time + (i * 3600)
            )
            episodes.append(episode)
            self.narrative_manager._store_episode(episode)
        
        # Generate narrative
        narrative = self.narrative_manager.generate_life_narrative(max_episodes=10)
        
        self.assertIsInstance(narrative, str)
        self.assertIn("Life Narrative", narrative)
        self.assertIn("Episode 1", narrative)
        self.assertIn("Episode 2", narrative)
        self.assertIn("Episode 3", narrative)
        self.assertIn("Life Summary", narrative)


class TestReflectiveReplayEngine(unittest.TestCase):
    """Test cases for ReflectiveReplayEngine."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.test_db.close()
        self.db_path = self.test_db.name
        
        self.reflective_engine = ReflectiveReplayEngine(self.db_path)
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_reflective_engine_initialization(self):
        """Test ReflectiveReplayEngine initialization."""
        self.assertIsNotNone(self.reflective_engine.reflection_depth_levels)
        self.assertIn('surface', self.reflective_engine.reflection_depth_levels)
        self.assertIn('moderate', self.reflective_engine.reflection_depth_levels)
        self.assertIn('deep', self.reflective_engine.reflection_depth_levels)
        
        self.assertIsNotNone(self.reflective_engine.reflection_triggers)
        self.assertIsNotNone(self.reflective_engine.trait_update_weights)
    
    def test_importance_to_text_conversion(self):
        """Test importance score to text conversion."""
        self.assertEqual(self.reflective_engine._importance_to_text(0.9), "highly significant")
        self.assertEqual(self.reflective_engine._importance_to_text(0.7), "moderately significant")
        self.assertEqual(self.reflective_engine._importance_to_text(0.5), "somewhat significant")
        self.assertEqual(self.reflective_engine._importance_to_text(0.2), "minor significance")
    
    def test_extract_primary_life_lesson(self):
        """Test life lesson extraction from episodes."""
        current_time = time.time()
        
        # Challenge episode with positive outcome
        challenge_episode = EpisodicMemory(
            episode_id="challenge_test",
            title="Challenge Episode",
            description="Overcoming difficulties",
            start_timestamp=current_time,
            end_timestamp=current_time + 3600,
            fact_ids=["fact1"],
            importance_score=0.7,
            emotional_valence=0.3,  # Positive outcome
            emotional_arousal=0.6,
            causal_impact=0.5,
            novelty_score=0.4,
            themes=["challenges"],
            reflection_notes="",
            identity_impact={},
            created_at=current_time,
            last_reflected=0.0
        )
        
        lesson = self.reflective_engine._extract_primary_life_lesson(challenge_episode)
        self.assertIn("challenge", lesson.lower())
        self.assertIn("resilience", lesson.lower())
        
        # Learning episode
        learning_episode = EpisodicMemory(
            episode_id="learning_test",
            title="Learning Episode",
            description="Acquiring new knowledge",
            start_timestamp=current_time,
            end_timestamp=current_time + 3600,
            fact_ids=["fact1"],
            importance_score=0.8,
            emotional_valence=0.4,
            emotional_arousal=0.5,
            causal_impact=0.6,
            novelty_score=0.7,
            themes=["learning"],
            reflection_notes="",
            identity_impact={},
            created_at=current_time,
            last_reflected=0.0
        )
        
        lesson = self.reflective_engine._extract_primary_life_lesson(learning_episode)
        self.assertIn("learning", lesson.lower())
        self.assertIn("growth", lesson.lower())
    
    def test_generate_reflection_insights(self):
        """Test reflection insight generation."""
        current_time = time.time()
        
        high_importance_episode = EpisodicMemory(
            episode_id="insight_test",
            title="High Impact Episode",
            description="Very significant experience",
            start_timestamp=current_time,
            end_timestamp=current_time + 3600,
            fact_ids=["fact1", "fact2"],
            importance_score=0.9,
            emotional_valence=0.6,
            emotional_arousal=0.8,
            causal_impact=0.7,
            novelty_score=0.8,
            themes=["learning", "achievements"],
            reflection_notes="",
            identity_impact={"curious": 0.3, "analytical": 0.2},
            created_at=current_time,
            last_reflected=0.0
        )
        
        insights = self.reflective_engine._generate_reflection_insights(high_importance_episode, 0.8)
        
        self.assertIsInstance(insights, list)
        self.assertGreater(len(insights), 0)
        
        # Check for expected insight types
        insight_text = " ".join(insights).lower()
        self.assertIn("significant", insight_text)  # High importance
        self.assertIn("positive", insight_text)     # Positive valence
        # High arousal should be mentioned in some form
        self.assertTrue(any(word in insight_text for word in ["intense", "intensity", "memorable", "impactful"]))
    
    def test_analyze_identity_implications(self):
        """Test identity implication analysis."""
        current_time = time.time()
        
        episode = EpisodicMemory(
            episode_id="identity_test",
            title="Identity Impact Episode",
            description="Episode affecting identity",
            start_timestamp=current_time,
            end_timestamp=current_time + 3600,
            fact_ids=["fact1"],
            importance_score=0.8,
            emotional_valence=0.4,
            emotional_arousal=0.6,
            causal_impact=0.5,
            novelty_score=0.6,
            themes=["learning", "problem-solving"],
            reflection_notes="",
            identity_impact={"curious": 0.2, "analytical": 0.1},
            created_at=current_time,
            last_reflected=0.0
        )
        
        implications = self.reflective_engine._analyze_identity_implications(episode, 0.8)
        
        self.assertIsInstance(implications, dict)
        self.assertIn("curious", implications)
        self.assertIn("analytical", implications)
        
        # Check that implications are reasonable
        for trait, impact in implications.items():
            self.assertLessEqual(impact, 0.5)
            self.assertGreaterEqual(impact, -0.2)
    
    def test_extract_emotional_learning(self):
        """Test emotional learning extraction."""
        current_time = time.time()
        
        emotional_episode = EpisodicMemory(
            episode_id="emotional_test",
            title="Emotional Episode",
            description="High emotion experience",
            start_timestamp=current_time,
            end_timestamp=current_time + 3600,
            fact_ids=["fact1"],
            importance_score=0.6,
            emotional_valence=-0.4,  # Negative
            emotional_arousal=0.8,   # High arousal
            causal_impact=0.6,
            novelty_score=0.5,
            themes=["emotion-frustrated", "challenges"],
            reflection_notes="",
            identity_impact={"resilient": 0.2},
            created_at=current_time,
            last_reflected=0.0
        )
        
        learning = self.reflective_engine._extract_emotional_learning(emotional_episode, 0.7)
        
        self.assertIsInstance(learning, list)
        self.assertGreater(len(learning), 0)
        
        # Check for expected learning content
        learning_text = " ".join(learning).lower()
        self.assertIn("frustrat", learning_text)  # Should mention frustration
    
    def test_identify_causal_patterns(self):
        """Test causal pattern identification."""
        current_time = time.time()
        
        causal_episode = EpisodicMemory(
            episode_id="causal_test",
            title="Causal Episode",
            description="High causal impact experience",
            start_timestamp=current_time,
            end_timestamp=current_time + 3600,
            fact_ids=["fact1"],
            importance_score=0.7,
            emotional_valence=0.3,
            emotional_arousal=0.6,
            causal_impact=0.8,  # High causal impact
            novelty_score=0.5,
            themes=["learning", "problem-solving"],
            reflection_notes="",
            identity_impact={"analytical": 0.2},
            created_at=current_time,
            last_reflected=0.0
        )
        
        patterns = self.reflective_engine._identify_causal_patterns(causal_episode, 0.8)
        
        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)
        
        # Check for causal language
        patterns_text = " ".join(patterns).lower()
        self.assertIn("causal", patterns_text)


class TestIntegration(unittest.TestCase):
    """Integration tests for life narrative system."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.test_db.close()
        self.db_path = self.test_db.name
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_narrative_manager_singleton(self):
        """Test that get_life_narrative_manager returns singleton."""
        manager1 = get_life_narrative_manager(self.db_path)
        manager2 = get_life_narrative_manager(self.db_path)
        
        # Should be the same instance
        self.assertIs(manager1, manager2)
    
    def test_reflective_engine_singleton(self):
        """Test that get_reflective_engine returns singleton."""
        engine1 = get_reflective_engine(self.db_path)
        engine2 = get_reflective_engine(self.db_path)
        
        # Should be the same instance
        self.assertIs(engine1, engine2)
    
    def test_end_to_end_episode_creation(self):
        """Test complete episode creation workflow."""
        
        # Create mock facts
        current_time = time.time()
        mock_facts = [
            EnhancedTripletFact(
                subject="system",
                predicate="learned",
                object="concept",
                timestamp=current_time,
                confidence=0.8,
                id="fact1",
                emotional_strength=0.6,
                emotion_tag="curiosity"
            ),
            EnhancedTripletFact(
                subject="system",
                predicate="solved",
                object="problem",
                timestamp=current_time + 600,
                confidence=0.9,
                id="fact2",
                causal_strength=0.7
            )
        ]
        
        narrative_manager = LifeNarrativeManager(self.db_path)
        
        # Test episode creation from facts
        episode = narrative_manager._create_episode_from_facts(mock_facts)
        
        self.assertIsNotNone(episode)
        self.assertIsInstance(episode, EpisodicMemory)
        self.assertEqual(len(episode.fact_ids), 2)
        self.assertIn("fact1", episode.fact_ids)
        self.assertIn("fact2", episode.fact_ids)
        self.assertGreater(episode.importance_score, 0)
        self.assertIn("learning", episode.themes)
    
    def test_episode_replay_workflow(self):
        """Test episode replay workflow."""
        current_time = time.time()
        
        # Create mock episode
        mock_episode = EpisodicMemory(
            episode_id="replay_test",
            title="Test Replay Episode",
            description="Testing replay functionality",
            start_timestamp=current_time,
            end_timestamp=current_time + 3600,
            fact_ids=["fact1", "fact2"],
            importance_score=0.7,
            emotional_valence=0.4,
            emotional_arousal=0.6,
            causal_impact=0.5,
            novelty_score=0.6,
            themes=["learning", "testing"],
            reflection_notes="Previous reflection",
            identity_impact={"curious": 0.2},
            created_at=current_time,
            last_reflected=current_time - 3600
        )
        
        # Test basic replay functionality without actual database integration
        # This is a simplified test to check the basic flow
        reflective_engine = ReflectiveReplayEngine(self.db_path)
        
        # Test replay with non-existent episode (should return error)
        replay_result = reflective_engine.replay_episode("nonexistent_episode")
        
        # Should gracefully handle missing episode
        self.assertIn('error', replay_result)


if __name__ == '__main__':
    # Run with increased verbosity to see test progress
    unittest.main(verbosity=2)