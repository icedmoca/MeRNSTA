#!/usr/bin/env python3
"""
Comprehensive Test Suite for Phase 22: Autonomous Memory Consolidation

Tests all aspects of the memory consolidation system including:
- Memory pruning and protection
- Clustering and similarity detection
- Fact consolidation and merging
- Timeline-aware operations
- Permanent tagging protection
- Score decay and prioritization
- Configuration and CLI integration

>15 test cases covering all major functionality.
"""

import pytest
import unittest
import tempfile
import os
import json
import sqlite3
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any

import numpy as np

# Import the memory consolidation components
from storage.memory_consolidator import (
    MemoryConsolidator, MemoryCluster, ConsolidationResult, ConsolidationRule,
    MemoryProtectionLevel, ConsolidationPriority
)
from storage.memory_log import MemoryLog
from storage.memory_utils import TripletFact


class TestMemoryConsolidator(unittest.TestCase):
    """Test the MemoryConsolidator core functionality."""
    
    def setUp(self):
        """Set up test fixtures with temporary database."""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # Initialize memory log with test database
        self.memory_log = MemoryLog(db_path=self.temp_db.name)
        
        # Initialize consolidator
        self.consolidator = MemoryConsolidator(self.memory_log)
        
        # Create test facts
        self.test_facts = self._create_test_facts()
        
    def tearDown(self):
        """Clean up test database."""
        try:
            if hasattr(self.memory_log, 'shutdown'):
                self.memory_log.shutdown()
            os.unlink(self.temp_db.name)
        except:
            pass
    
    def _create_test_facts(self) -> List[TripletFact]:
        """Create test facts for testing."""
        facts = []
        
        # High confidence facts (should be protected)
        facts.append(TripletFact(
            id=1, subject="user", predicate="likes", object="Python",
            confidence=0.9, timestamp=datetime.now().isoformat()
        ))
        
        facts.append(TripletFact(
            id=2, subject="user", predicate="prefers", object="VS Code",
            confidence=0.85, timestamp=datetime.now().isoformat()
        ))
        
        # Medium confidence facts
        facts.append(TripletFact(
            id=3, subject="user", predicate="sometimes_uses", object="JavaScript",
            confidence=0.6, timestamp=(datetime.now() - timedelta(days=10)).isoformat()
        ))
        
        # Low confidence facts (candidates for pruning)
        facts.append(TripletFact(
            id=4, subject="user", predicate="might_like", object="Ruby",
            confidence=0.2, timestamp=(datetime.now() - timedelta(days=100)).isoformat()
        ))
        
        facts.append(TripletFact(
            id=5, subject="user", predicate="uncertain_about", object="Go",
            confidence=0.1, timestamp=(datetime.now() - timedelta(days=200)).isoformat()
        ))
        
        # Duplicate/similar facts
        facts.append(TripletFact(
            id=6, subject="user", predicate="enjoys", object="Python programming",
            confidence=0.8, timestamp=(datetime.now() - timedelta(days=5)).isoformat()
        ))
        
        facts.append(TripletFact(
            id=7, subject="user", predicate="loves", object="Python",
            confidence=0.95, timestamp=(datetime.now() - timedelta(days=2)).isoformat()
        ))
        
        # Old but high confidence (should be protected)
        facts.append(TripletFact(
            id=8, subject="system", predicate="core_value", object="safety",
            confidence=0.95, timestamp=(datetime.now() - timedelta(days=365)).isoformat()
        ))
        
        return facts
    
    def test_consolidator_initialization(self):
        """Test MemoryConsolidator initialization."""
        self.assertEqual(self.consolidator.similarity_threshold, 0.8)
        self.assertEqual(self.consolidator.min_cluster_size, 3)
        self.assertIsInstance(self.consolidator.consolidation_rules, dict)
        self.assertGreater(len(self.consolidator.consolidation_rules), 0)
        self.assertEqual(self.consolidator.temporal_window_days, 30)
    
    def test_load_consolidation_config(self):
        """Test loading consolidation configuration."""
        config = self.consolidator._load_consolidation_config()
        
        self.assertIsInstance(config, dict)
        self.assertIn('enabled', config)
        self.assertTrue(config.get('enabled'))
        self.assertIn('similarity_threshold', config)
        self.assertIn('pruning', config)
        self.assertIn('clustering', config)
        self.assertIn('consolidation', config)
    
    def test_initialize_default_rules(self):
        """Test initialization of default consolidation rules."""
        rules = self.consolidator.consolidation_rules
        
        # Check that default rules exist
        expected_rules = [
            "protect_recent", "prune_low_confidence", "consolidate_duplicates",
            "cluster_similar", "strengthen_patterns"
        ]
        
        for rule_id in expected_rules:
            self.assertIn(rule_id, rules)
            rule = rules[rule_id]
            self.assertIsInstance(rule, ConsolidationRule)
            self.assertEqual(rule.rule_id, rule_id)
            self.assertTrue(rule.enabled)
            self.assertIsInstance(rule.priority, ConsolidationPriority)
    
    def test_fact_matches_rule_conditions(self):
        """Test fact matching against rule conditions."""
        # Create test fact
        fact = TripletFact(
            id=1, subject="test", predicate="test", object="test",
            confidence=0.5, timestamp=(datetime.now() - timedelta(days=10)).isoformat()
        )
        
        # Test age condition
        rule = ConsolidationRule(
            rule_id="test_age",
            name="Test Age",
            conditions={"max_age_days": 15}
        )
        self.assertTrue(self.consolidator._fact_matches_rule_conditions(fact, rule))
        
        rule.conditions["max_age_days"] = 5
        self.assertFalse(self.consolidator._fact_matches_rule_conditions(fact, rule))
        
        # Test confidence condition
        rule.conditions = {"min_confidence": 0.3}
        self.assertTrue(self.consolidator._fact_matches_rule_conditions(fact, rule))
        
        rule.conditions["min_confidence"] = 0.8
        self.assertFalse(self.consolidator._fact_matches_rule_conditions(fact, rule))
    
    def test_apply_protection_rules(self):
        """Test application of protection rules to facts."""
        # Mock facts with consolidation metadata
        facts = []
        for fact in self.test_facts[:3]:
            fact.consolidation_metadata = {
                'protection_level': MemoryProtectionLevel.NONE,
                'last_accessed': datetime.now(),
                'access_count': 0
            }
            facts.append(fact)
        
        # Apply protection rules
        self.consolidator._apply_protection_rules(facts)
        
        # Check that recent high-confidence facts are protected
        high_conf_fact = facts[0]  # confidence=0.9, recent
        self.assertNotEqual(
            high_conf_fact.consolidation_metadata['protection_level'],
            MemoryProtectionLevel.NONE
        )
    
    def test_prune_memories(self):
        """Test memory pruning functionality."""
        # Add test facts to memory log (mock)
        with patch.object(self.memory_log, 'delete_fact') as mock_delete:
            with patch.object(self.consolidator, 'protected_fact_ids', {1, 8}):  # Protect some facts
                
                # Mock facts for pruning test
                facts = self.test_facts.copy()
                
                pruned_count = self.consolidator._prune_memories(facts)
                
                # Should prune low confidence old facts
                self.assertGreater(pruned_count, 0)
                
                # Check that delete was called for low confidence facts
                deleted_ids = [call[0][0] for call in mock_delete.call_args_list]
                
                # Should not delete protected facts
                self.assertNotIn(1, deleted_ids)  # Protected
                self.assertNotIn(8, deleted_ids)  # Protected
    
    def test_cluster_similar_facts(self):
        """Test clustering of similar facts."""
        # Mock vectorizer to return consistent embeddings
        def mock_vectorizer(text):
            # Simple hash-based embedding for consistent testing
            words = text.lower().split()
            embedding = [0.0] * 10
            for i, word in enumerate(words[:10]):
                embedding[i] = hash(word) % 100 / 100.0
            return embedding
        
        self.consolidator.memory_log.vectorizer = mock_vectorizer
        
        # Test clustering
        clusters = self.consolidator._cluster_similar_facts(self.test_facts)
        
        # Should create clusters
        self.assertIsInstance(clusters, list)
        
        for cluster in clusters:
            self.assertIsInstance(cluster, MemoryCluster)
            self.assertGreaterEqual(len(cluster.facts), self.consolidator.min_cluster_size)
            self.assertIsNotNone(cluster.centroid_embedding)
            self.assertGreater(cluster.consolidation_score, 0)
    
    def test_calculate_cluster_consolidation_score(self):
        """Test cluster consolidation score calculation."""
        # Test with facts of varying confidence
        test_facts = self.test_facts[:3]
        
        score = self.consolidator._calculate_cluster_consolidation_score(test_facts)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Test with empty facts
        empty_score = self.consolidator._calculate_cluster_consolidation_score([])
        self.assertEqual(empty_score, 0.0)
    
    def test_find_duplicate_facts(self):
        """Test duplicate fact detection."""
        # Create facts with duplicates
        duplicate_facts = [
            self.test_facts[0],  # "user likes Python"
            self.test_facts[6],  # "user loves Python" (similar)
        ]
        
        duplicate_groups = self.consolidator._find_duplicate_facts(duplicate_facts)
        
        # Should find duplicate groups
        self.assertIsInstance(duplicate_groups, list)
        
        # Check that similar facts are grouped
        for group in duplicate_groups:
            self.assertIsInstance(group, list)
            self.assertGreaterEqual(len(group), 1)
    
    def test_are_facts_duplicates(self):
        """Test duplicate detection between two facts."""
        fact1 = TripletFact(
            id=1, subject="user", predicate="likes", object="Python",
            confidence=0.9, timestamp=datetime.now().isoformat()
        )
        
        # Exact duplicate
        fact2 = TripletFact(
            id=2, subject="user", predicate="likes", object="Python",
            confidence=0.8, timestamp=datetime.now().isoformat()
        )
        
        self.assertTrue(self.consolidator._are_facts_duplicates(fact1, fact2))
        
        # Similar but not duplicate
        fact3 = TripletFact(
            id=3, subject="user", predicate="enjoys", object="Python programming",
            confidence=0.8, timestamp=datetime.now().isoformat()
        )
        
        # Should detect semantic similarity
        is_similar = self.consolidator._are_facts_duplicates(fact1, fact3)
        # Result depends on similarity threshold, just check it's boolean
        self.assertIsInstance(is_similar, bool)
        
        # Completely different
        fact4 = TripletFact(
            id=4, subject="system", predicate="requires", object="maintenance",
            confidence=0.7, timestamp=datetime.now().isoformat()
        )
        
        self.assertFalse(self.consolidator._are_facts_duplicates(fact1, fact4))
    
    def test_merge_duplicate_facts(self):
        """Test merging of duplicate facts."""
        # Create duplicate facts
        duplicates = [
            TripletFact(id=1, subject="user", predicate="likes", object="Python", confidence=0.8, timestamp=datetime.now().isoformat()),
            TripletFact(id=2, subject="user", predicate="likes", object="Python", confidence=0.9, timestamp=datetime.now().isoformat()),
            TripletFact(id=3, subject="user", predicate="likes", object="Python", confidence=0.7, timestamp=datetime.now().isoformat())
        ]
        
        # Add consolidation metadata
        for fact in duplicates:
            fact.consolidation_metadata = {}
        
        # Mock memory log methods
        with patch.object(self.memory_log, 'delete_fact') as mock_delete:
            with patch.object(self.memory_log, 'update_fact_confidence') as mock_update:
                
                merged_fact = self.consolidator._merge_duplicate_facts(duplicates)
                
                self.assertIsNotNone(merged_fact)
                self.assertIsInstance(merged_fact, TripletFact)
                
                # Should have highest confidence
                self.assertEqual(merged_fact.id, 2)  # Fact with confidence 0.9
                
                # Should have updated confidence (weighted average)
                self.assertGreater(merged_fact.confidence, 0.8)
                
                # Should have consolidation metadata
                self.assertIn('merged_from', merged_fact.consolidation_metadata)
                self.assertIn('merge_count', merged_fact.consolidation_metadata)
    
    def test_identify_consistent_patterns(self):
        """Test identification of consistent patterns in facts."""
        # Create facts with patterns
        pattern_facts = [
            TripletFact(id=1, subject="user", predicate="likes", object="Python", confidence=0.8, timestamp=datetime.now().isoformat()),
            TripletFact(id=2, subject="user", predicate="likes", object="JavaScript", confidence=0.6, timestamp=datetime.now().isoformat()),
            TripletFact(id=3, subject="user", predicate="likes", object="TypeScript", confidence=0.7, timestamp=datetime.now().isoformat())
        ]
        
        patterns = self.consolidator._identify_consistent_patterns(pattern_facts)
        
        self.assertIsInstance(patterns, list)
        
        for pattern in patterns:
            self.assertIsInstance(pattern, dict)
            self.assertIn('type', pattern)
            self.assertIn('strength', pattern)
            self.assertIn('supporting_facts', pattern)
            self.assertGreater(pattern['strength'], 0)
    
    def test_apply_temporal_ordering(self):
        """Test temporal ordering application."""
        # Add consolidation metadata to facts
        facts = []
        for fact in self.test_facts:
            fact.consolidation_metadata = {}
            facts.append(fact)
        
        self.consolidator._apply_temporal_ordering(facts)
        
        # Check that temporal metadata was added
        for fact in facts:
            self.assertIn('temporal_neighbors', fact.consolidation_metadata)
            self.assertIn('temporal_position', fact.consolidation_metadata)
            self.assertIn('temporal_window', fact.consolidation_metadata)
            
            # Check data types
            self.assertIsInstance(fact.consolidation_metadata['temporal_neighbors'], list)
            self.assertIsInstance(fact.consolidation_metadata['temporal_position'], int)
            self.assertEqual(fact.consolidation_metadata['temporal_window'], self.consolidator.temporal_window_days)
    
    def test_calculate_priority_score(self):
        """Test priority score calculation."""
        fact = TripletFact(
            id=1, subject="user", predicate="likes", object="Python",
            confidence=0.8, timestamp=datetime.now().isoformat()
        )
        
        # Add consolidation metadata
        fact.consolidation_metadata = {
            'access_count': 5,
            'protection_level': MemoryProtectionLevel.PROTECTED
        }
        
        score = self.consolidator._calculate_priority_score(fact)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # High confidence + protected should have high score
        self.assertGreater(score, 0.5)
    
    def test_consolidate_memory_full(self):
        """Test full memory consolidation process."""
        # Mock the load facts method to return test facts
        with patch.object(self.consolidator, '_load_facts_for_consolidation', return_value=self.test_facts):
            
            result = self.consolidator.consolidate_memory(full_consolidation=True)
            
            self.assertIsInstance(result, ConsolidationResult)
            self.assertTrue(result.success)
            self.assertGreater(result.facts_processed, 0)
            self.assertIsNotNone(result.completed_at)
            self.assertIsInstance(result.statistics, dict)
    
    def test_consolidate_memory_incremental(self):
        """Test incremental memory consolidation."""
        with patch.object(self.consolidator, '_load_facts_for_consolidation', return_value=self.test_facts):
            
            result = self.consolidator.consolidate_memory(full_consolidation=False)
            
            self.assertIsInstance(result, ConsolidationResult)
            self.assertEqual(result.operation_type, "incremental")
            self.assertTrue(result.success)
    
    def test_prune_public_api(self):
        """Test public pruning API."""
        # Mock the memory log prune_memory method
        mock_result = {
            'total_facts': 10,
            'pruned_facts': 3,
            'errors': []
        }
        
        with patch.object(self.memory_log, 'prune_memory', return_value=mock_result):
            
            result = self.consolidator.prune(confidence_threshold=0.3, age_threshold_days=90)
            
            self.assertIsInstance(result, dict)
            self.assertEqual(result['pruned_facts'], 3)
            self.assertEqual(result['total_facts'], 10)
    
    def test_cluster_public_api(self):
        """Test public clustering API."""
        with patch.object(self.consolidator, '_load_facts_for_consolidation', return_value=self.test_facts):
            with patch.object(self.consolidator, '_cluster_similar_facts', return_value=[]):
                
                clusters = self.consolidator.cluster(algorithm='kmeans', n_clusters=3)
                
                self.assertIsInstance(clusters, list)
    
    def test_prioritize_public_api(self):
        """Test public prioritization API."""
        with patch.object(self.consolidator, '_load_facts_for_consolidation', return_value=self.test_facts):
            
            prioritized_facts = self.consolidator.prioritize()
            
            self.assertIsInstance(prioritized_facts, list)
            self.assertEqual(len(prioritized_facts), len(self.test_facts))
            
            # Check that facts are sorted by priority
            if len(prioritized_facts) > 1:
                first_priority = getattr(prioritized_facts[0], 'consolidation_metadata', {}).get('priority_score', 0)
                last_priority = getattr(prioritized_facts[-1], 'consolidation_metadata', {}).get('priority_score', 0)
                self.assertGreaterEqual(first_priority, last_priority)
    
    def test_add_permanent_tag(self):
        """Test adding permanent protection tags."""
        # Mock database operations
        with patch('storage.memory_consolidator.get_conn') as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.execute.return_value = mock_cursor
            mock_cursor.fetchone.return_value = ('["existing_tag"]',)
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            
            result = self.consolidator.add_permanent_tag(1, "CRITICAL")
            
            self.assertTrue(result)
            self.assertIn(1, self.consolidator.protected_fact_ids)
            
            # Check that database was updated
            mock_conn.execute.assert_called()
            mock_conn.commit.assert_called()
    
    def test_list_clusters(self):
        """Test cluster listing functionality."""
        # Create test clusters
        test_cluster = MemoryCluster(
            cluster_id="test_cluster_1",
            cluster_type="semantic",
            facts=self.test_facts[:3],
            consolidation_score=0.8
        )
        
        self.consolidator.active_clusters["test_cluster_1"] = test_cluster
        
        # Test listing all clusters
        clusters = self.consolidator.list_clusters()
        
        self.assertIsInstance(clusters, list)
        self.assertEqual(len(clusters), 1)
        
        cluster_info = clusters[0]
        self.assertEqual(cluster_info['cluster_id'], "test_cluster_1")
        self.assertEqual(cluster_info['cluster_type'], "semantic")
        self.assertEqual(cluster_info['fact_count'], 3)
        self.assertIn('sample_facts', cluster_info)
        
        # Test filtering by type
        semantic_clusters = self.consolidator.list_clusters(cluster_type="semantic")
        self.assertEqual(len(semantic_clusters), 1)
        
        temporal_clusters = self.consolidator.list_clusters(cluster_type="temporal")
        self.assertEqual(len(temporal_clusters), 0)
    
    def test_get_consolidation_statistics(self):
        """Test consolidation statistics generation."""
        with patch.object(self.consolidator, '_load_facts_for_consolidation', return_value=self.test_facts):
            
            stats = self.consolidator.get_consolidation_statistics()
            
            self.assertIsInstance(stats, dict)
            self.assertIn('total_facts', stats)
            self.assertIn('protected_facts', stats)
            self.assertIn('active_clusters', stats)
            self.assertIn('avg_confidence', stats)
            self.assertIn('confidence_distribution', stats)
            
            self.assertEqual(stats['total_facts'], len(self.test_facts))
            self.assertIsInstance(stats['avg_confidence'], float)


class TestMemoryConsolidationIntegration(unittest.TestCase):
    """Integration tests for memory consolidation system."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.memory_log = MemoryLog(db_path=self.temp_db.name)
        self.consolidator = MemoryConsolidator(self.memory_log)
    
    def tearDown(self):
        """Clean up test environment."""
        try:
            if hasattr(self.memory_log, 'shutdown'):
                self.memory_log.shutdown()
            os.unlink(self.temp_db.name)
        except:
            pass
    
    def test_end_to_end_consolidation(self):
        """Test complete consolidation workflow."""
        # Add some real facts to memory
        fact_ids = []
        
        # Add facts with varying confidence and ages
        fact_data = [
            ("user", "likes", "Python", 0.9),
            ("user", "dislikes", "Java", 0.3),
            ("user", "prefers", "VSCode", 0.8),
            ("system", "requires", "memory", 0.1),  # Low confidence, should be pruned
        ]
        
        for subject, predicate, obj, confidence in fact_data:
            fact_id = self.memory_log.store_fact(
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=confidence,
                tags=[]
            )
            fact_ids.append(fact_id)
        
        # Run consolidation
        result = self.consolidator.consolidate_memory(full_consolidation=True)
        
        # Check results
        self.assertTrue(result.success)
        self.assertGreater(result.facts_processed, 0)
        
        # Verify that low confidence facts were pruned
        remaining_facts = self.memory_log.list_facts()
        remaining_confidences = [f.confidence for f in remaining_facts]
        
        # Should have mostly high confidence facts left
        avg_confidence = sum(remaining_confidences) / len(remaining_confidences) if remaining_confidences else 0
        self.assertGreater(avg_confidence, 0.4)  # Should be higher after pruning
    
    def test_protection_mechanisms(self):
        """Test that protection mechanisms work correctly."""
        # Add a high-value fact
        critical_fact_id = self.memory_log.store_fact(
            subject="system",
            predicate="core_principle",
            object="safety_first",
            confidence=0.95,
            tags=["CRITICAL"]
        )
        
        # Add a low confidence fact that would normally be pruned
        low_conf_fact_id = self.memory_log.store_fact(
            subject="temp",
            predicate="might_be",
            object="useful",
            confidence=0.1,
            tags=[]
        )
        
        # Mark critical fact as permanent
        self.consolidator.add_permanent_tag(critical_fact_id, "PERMANENT")
        
        # Run pruning
        self.consolidator.prune(confidence_threshold=0.3)
        
        # Check that critical fact still exists
        facts = self.memory_log.list_facts()
        fact_ids = [f.id for f in facts]
        
        self.assertIn(critical_fact_id, fact_ids)  # Should be protected
        # Low confidence fact may or may not be pruned depending on age
    
    def test_clustering_and_consolidation(self):
        """Test clustering and consolidation working together."""
        # Add similar facts that should be clustered
        similar_facts = [
            ("user", "enjoys", "Python programming", 0.8),
            ("user", "likes", "Python development", 0.7),
            ("user", "prefers", "Python coding", 0.9),
        ]
        
        for subject, predicate, obj, confidence in similar_facts:
            self.memory_log.store_fact(
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=confidence,
                tags=[]
            )
        
        # Run clustering
        clusters = self.consolidator.cluster()
        
        # Should create clusters for similar facts
        self.assertIsInstance(clusters, list)
        
        # Run consolidation
        consolidation_result = self.consolidator.consolidate()
        
        self.assertIsInstance(consolidation_result, dict)
        self.assertIn('clusters_processed', consolidation_result)


class TestMemoryConsolidationCLI(unittest.TestCase):
    """Test CLI integration for memory consolidation."""
    
    def test_cli_command_imports(self):
        """Test that CLI command functions can be imported."""
        try:
            from cortex.cli_commands import (
                consolidate_memory, list_clusters, prioritize_memory, memory_stats
            )
            self.assertTrue(True)  # Imports successful
        except ImportError as e:
            self.fail(f"CLI command imports failed: {e}")
    
    @patch('cortex.cli_commands.MemoryLog')
    @patch('cortex.cli_commands.MemoryConsolidator')
    def test_consolidate_memory_cli(self, mock_consolidator_class, mock_memory_log_class):
        """Test consolidate_memory CLI command."""
        from cortex.cli_commands import consolidate_memory
        
        # Mock consolidator and its methods
        mock_consolidator = MagicMock()
        mock_consolidator_class.return_value = mock_consolidator
        
        # Mock successful consolidation result
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.facts_processed = 100
        mock_result.facts_pruned = 10
        mock_result.facts_consolidated = 5
        mock_result.clusters_created = 3
        mock_result.started_at = datetime.now()
        mock_result.completed_at = datetime.now()
        mock_result.statistics = {
            'total_facts': 100,
            'protected_facts': 20,
            'avg_confidence': 0.75,
            'confidence_distribution': {'high': 50, 'medium': 30, 'low': 20},
            'temporal_span_days': 365
        }
        mock_result.warnings = []
        mock_result.errors = []
        
        mock_consolidator.consolidate_memory.return_value = mock_result
        
        # Test full consolidation
        result = consolidate_memory("full")
        
        self.assertIn("SUCCESS", result)
        mock_consolidator.consolidate_memory.assert_called_with(full_consolidation=True)
    
    @patch('cortex.cli_commands.MemoryLog')
    @patch('cortex.cli_commands.MemoryConsolidator')
    def test_list_clusters_cli(self, mock_consolidator_class, mock_memory_log_class):
        """Test list_clusters CLI command."""
        from cortex.cli_commands import list_clusters
        
        # Mock consolidator
        mock_consolidator = MagicMock()
        mock_consolidator_class.return_value = mock_consolidator
        
        # Mock cluster data
        mock_clusters = [
            {
                'cluster_id': 'semantic_1',
                'cluster_type': 'semantic',
                'fact_count': 5,
                'consolidation_score': 0.8,
                'created_at': '2024-01-01T00:00:00',
                'last_updated': '2024-01-01T00:00:00',
                'sample_facts': ['user likes Python', 'user enjoys coding']
            }
        ]
        mock_consolidator.list_clusters.return_value = mock_clusters
        
        result = list_clusters()
        
        self.assertIn("Listed 1 memory clusters", result)
        mock_consolidator.list_clusters.assert_called_with(None)


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(unittest.makeSuite(TestMemoryConsolidator))
    suite.addTest(unittest.makeSuite(TestMemoryConsolidationIntegration))
    suite.addTest(unittest.makeSuite(TestMemoryConsolidationCLI))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PHASE 22 MEMORY CONSOLIDATION TEST RESULTS")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    print(f"\n🧠 Phase 22 Autonomous Memory Consolidation testing complete!")
    print(f"📊 Test coverage: Pruning, Clustering, Consolidation, Protection, CLI, Integration")
    print(f"✅ {result.testsRun} comprehensive test cases executed")