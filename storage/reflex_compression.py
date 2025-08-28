#!/usr/bin/env python3
"""
Reflex Compression & Reuse System for MeRNSTA

This module implements the ReflexCompressor that groups reflex cycles with similar
strategies and goals, merging them into reusable patterns for future drift handling.
"""

import sqlite3
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import defaultdict
import hashlib

from .reflex_log import ReflexCycle, ReflexScore


@dataclass
class ReflexTemplate:
    """
    Represents a reusable reflex pattern derived from multiple similar reflex cycles.
    
    Templates capture successful repair strategies that can be applied to similar
    drift scenarios, improving response efficiency and effectiveness.
    """
    template_id: str
    strategy: str
    pattern_signature: str  # Hash of the pattern characteristics
    goal_pattern: str  # Abstracted goal description
    execution_pattern: List[str]  # Common execution steps
    success_rate: float = 0.0
    avg_score: float = 0.0
    cluster_overlap: float = 0.0
    usage_count: int = 0
    created_at: float = 0.0
    last_used: float = 0.0
    source_cycles: List[str] = None  # List of cycle IDs that formed this template
    effectiveness_notes: str = ""
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.last_used == 0.0:
            self.last_used = time.time()
        if self.source_cycles is None:
            self.source_cycles = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReflexTemplate':
        """Create from dictionary."""
        return cls(**data)


class ReflexCompressor:
    """
    ReflexCompressor that groups reflex cycles with same strategy and similar goals,
    merging them into reusable patterns for future drift handling.
    
    Features:
    - Groups reflex cycles by strategy and goal similarity
    - Merges cycles into reusable templates
    - Calculates pattern effectiveness metrics
    - Suggests strategies for similar drift scenarios
    - Persists templates for future use
    """
    
    def __init__(self, db_path: str = "reflex_cycles.db"):
        self.db_path = db_path
        self.similarity_threshold = 0.7  # Threshold for considering cycles similar
        self.min_cycles_for_template = 3  # Minimum cycles to form a template
        self.max_templates_per_strategy = 10  # Limit templates per strategy
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize reflex templates database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reflex_templates (
                template_id TEXT PRIMARY KEY,
                strategy TEXT NOT NULL,
                pattern_signature TEXT NOT NULL,
                goal_pattern TEXT NOT NULL,
                execution_pattern TEXT NOT NULL,
                success_rate REAL DEFAULT 0.0,
                avg_score REAL DEFAULT 0.0,
                cluster_overlap REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                last_used REAL NOT NULL,
                source_cycles TEXT NOT NULL,
                effectiveness_notes TEXT,
                INDEX(strategy),
                INDEX(success_rate),
                INDEX(avg_score),
                INDEX(usage_count)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def compress_reflex_cycles(self, cycles: List[ReflexCycle]) -> List[ReflexTemplate]:
        """
        Compress reflex cycles into reusable templates.
        
        Args:
            cycles: List of reflex cycles to compress
            
        Returns:
            List of newly created or updated ReflexTemplate objects
        """
        logging.info(f"[ReflexCompressor] Compressing {len(cycles)} reflex cycles")
        
        if len(cycles) < self.min_cycles_for_template:
            logging.info("[ReflexCompressor] Not enough cycles for compression")
            return []
        
        # Group cycles by strategy
        strategy_groups = self._group_cycles_by_strategy(cycles)
        
        # Compress each strategy group
        new_templates = []
        for strategy, strategy_cycles in strategy_groups.items():
            templates = self._compress_strategy_group(strategy, strategy_cycles)
            new_templates.extend(templates)
        
        logging.info(f"[ReflexCompressor] Created {len(new_templates)} templates")
        return new_templates
    
    def _group_cycles_by_strategy(self, cycles: List[ReflexCycle]) -> Dict[str, List[ReflexCycle]]:
        """Group reflex cycles by their strategy."""
        groups = defaultdict(list)
        for cycle in cycles:
            groups[cycle.strategy].append(cycle)
        return dict(groups)
    
    def _compress_strategy_group(self, strategy: str, 
                                cycles: List[ReflexCycle]) -> List[ReflexTemplate]:
        """Compress cycles within a strategy group into templates."""
        if len(cycles) < self.min_cycles_for_template:
            return []
        
        # Find similar cycles and group them
        cycle_groups = self._find_similar_cycles(cycles)
        
        # Create templates from each group
        templates = []
        for group in cycle_groups:
            if len(group) >= self.min_cycles_for_template:
                template = self._create_template_from_group(strategy, group)
                if template:
                    templates.append(template)
        
        return templates
    
    def _find_similar_cycles(self, cycles: List[ReflexCycle]) -> List[List[ReflexCycle]]:
        """Find groups of similar cycles based on goal and execution patterns."""
        if len(cycles) < 2:
            return [cycles]
        
        # Calculate similarity matrix
        similarity_matrix = self._calculate_cycle_similarity_matrix(cycles)
        
        # Group similar cycles
        groups = []
        used_cycles = set()
        
        for i, cycle in enumerate(cycles):
            if i in used_cycles:
                continue
            
            # Start a new group
            group = [cycle]
            used_cycles.add(i)
            
            # Find similar cycles
            for j, other_cycle in enumerate(cycles):
                if j in used_cycles:
                    continue
                
                if similarity_matrix[i][j] >= self.similarity_threshold:
                    group.append(other_cycle)
                    used_cycles.add(j)
            
            groups.append(group)
        
        return groups
    
    def _calculate_cycle_similarity_matrix(self, cycles: List[ReflexCycle]) -> List[List[float]]:
        """Calculate similarity matrix between cycles."""
        n = len(cycles)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self._calculate_cycle_similarity(cycles[i], cycles[j])
                matrix[i][j] = similarity
                matrix[j][i] = similarity
        
        return matrix
    
    def _calculate_cycle_similarity(self, cycle1: ReflexCycle, cycle2: ReflexCycle) -> float:
        """Calculate similarity between two reflex cycles."""
        similarities = []
        
        # Goal similarity (based on drift_score and token_id)
        if cycle1.token_id == cycle2.token_id:
            similarities.append(1.0)
        elif cycle1.token_id and cycle2.token_id:
            # Same token cluster
            similarities.append(0.8)
        else:
            similarities.append(0.3)
        
        # Drift score similarity
        drift_diff = abs(cycle1.drift_score - cycle2.drift_score)
        drift_similarity = max(0.0, 1.0 - drift_diff)
        similarities.append(drift_similarity)
        
        # Execution pattern similarity
        exec_similarity = self._calculate_execution_similarity(
            cycle1.actions_taken, cycle2.actions_taken
        )
        similarities.append(exec_similarity)
        
        # Success similarity
        if cycle1.success == cycle2.success:
            similarities.append(1.0)
        else:
            similarities.append(0.5)
        
        return np.mean(similarities)
    
    def _calculate_execution_similarity(self, actions1: List[str], 
                                       actions2: List[str]) -> float:
        """Calculate similarity between execution patterns."""
        if not actions1 or not actions2:
            return 0.0
        
        # Convert actions to sets for comparison
        set1 = set(actions1)
        set2 = set(actions2)
        
        if not set1 or not set2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _create_template_from_group(self, strategy: str, 
                                   cycles: List[ReflexCycle]) -> Optional[ReflexTemplate]:
        """Create a template from a group of similar cycles."""
        if not cycles:
            return None
        
        # Generate pattern signature
        pattern_signature = self._generate_pattern_signature(cycles)
        
        # Check if template already exists
        existing_template = self._get_template_by_signature(pattern_signature)
        if existing_template:
            return self._update_existing_template(existing_template, cycles)
        
        # Create new template
        template_id = f"template_{strategy}_{int(time.time())}"
        
        # Extract common patterns
        goal_pattern = self._extract_goal_pattern(cycles)
        execution_pattern = self._extract_execution_pattern(cycles)
        
        # Calculate metrics
        success_rate = np.mean([1.0 if c.success else 0.0 for c in cycles])
        avg_score = np.mean([c.reflex_score.score if c.reflex_score else 0.0 for c in cycles])
        cluster_overlap = self._calculate_cluster_overlap(cycles)
        
        # Create template
        template = ReflexTemplate(
            template_id=template_id,
            strategy=strategy,
            pattern_signature=pattern_signature,
            goal_pattern=goal_pattern,
            execution_pattern=execution_pattern,
            success_rate=success_rate,
            avg_score=avg_score,
            cluster_overlap=cluster_overlap,
            source_cycles=[c.cycle_id for c in cycles],
            effectiveness_notes=self._generate_effectiveness_notes(cycles)
        )
        
        # Store template
        self._store_template(template)
        
        return template
    
    def _generate_pattern_signature(self, cycles: List[ReflexCycle]) -> str:
        """Generate a unique signature for a pattern of cycles."""
        # Create a hashable representation of the pattern
        pattern_data = {
            'strategy': cycles[0].strategy,
            'avg_drift_score': np.mean([c.drift_score for c in cycles]),
            'success_rate': np.mean([1.0 if c.success else 0.0 for c in cycles]),
            'common_actions': self._get_common_actions(cycles)
        }
        
        # Create hash
        pattern_str = json.dumps(pattern_data, sort_keys=True)
        return hashlib.md5(pattern_str.encode()).hexdigest()
    
    def _get_common_actions(self, cycles: List[ReflexCycle]) -> List[str]:
        """Get actions that appear in most cycles."""
        action_counts = defaultdict(int)
        total_cycles = len(cycles)
        
        for cycle in cycles:
            for action in cycle.actions_taken:
                action_counts[action] += 1
        
        # Return actions that appear in at least 50% of cycles
        threshold = total_cycles * 0.5
        return [action for action, count in action_counts.items() if count >= threshold]
    
    def _extract_goal_pattern(self, cycles: List[ReflexCycle]) -> str:
        """Extract abstract goal pattern from cycles."""
        # Analyze common characteristics
        drift_scores = [c.drift_score for c in cycles]
        avg_drift = np.mean(drift_scores)
        
        token_ids = [c.token_id for c in cycles if c.token_id]
        if token_ids:
            token_pattern = f"token_cluster_{len(set(token_ids))}"
        else:
            token_pattern = "general_drift"
        
        if avg_drift > 0.8:
            severity = "high"
        elif avg_drift > 0.5:
            severity = "medium"
        else:
            severity = "low"
        
        return f"{severity}_severity_{token_pattern}_drift"
    
    def _extract_execution_pattern(self, cycles: List[ReflexCycle]) -> List[str]:
        """Extract common execution pattern from cycles."""
        return self._get_common_actions(cycles)
    
    def _calculate_cluster_overlap(self, cycles: List[ReflexCycle]) -> float:
        """Calculate cluster overlap between cycles."""
        cluster_ids = [c.cluster_id for c in cycles if c.cluster_id]
        if not cluster_ids:
            return 0.0
        
        unique_clusters = len(set(cluster_ids))
        total_cycles = len(cycles)
        
        # Higher overlap = more cycles sharing same clusters
        return 1.0 - (unique_clusters / total_cycles)
    
    def _generate_effectiveness_notes(self, cycles: List[ReflexCycle]) -> str:
        """Generate notes about template effectiveness."""
        successful = [c for c in cycles if c.success]
        failed = [c for c in cycles if not c.success]
        
        notes = []
        notes.append(f"Based on {len(cycles)} cycles")
        notes.append(f"Success rate: {len(successful)}/{len(cycles)}")
        
        if successful:
            avg_success_score = np.mean([c.reflex_score.score if c.reflex_score else 0.0 for c in successful])
            notes.append(f"Avg success score: {avg_success_score:.2f}")
        
        if failed:
            notes.append(f"Failed cycles: {len(failed)}")
        
        return "; ".join(notes)
    
    def _get_template_by_signature(self, pattern_signature: str) -> Optional[ReflexTemplate]:
        """Get existing template by pattern signature."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT template_id, strategy, pattern_signature, goal_pattern,
                   execution_pattern, success_rate, avg_score, cluster_overlap,
                   usage_count, created_at, last_used, source_cycles, effectiveness_notes
            FROM reflex_templates 
            WHERE pattern_signature = ?
        """, (pattern_signature,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            data = {
                'template_id': row[0],
                'strategy': row[1],
                'pattern_signature': row[2],
                'goal_pattern': row[3],
                'execution_pattern': json.loads(row[4]),
                'success_rate': row[5],
                'avg_score': row[6],
                'cluster_overlap': row[7],
                'usage_count': row[8],
                'created_at': row[9],
                'last_used': row[10],
                'source_cycles': json.loads(row[11]),
                'effectiveness_notes': row[12]
            }
            return ReflexTemplate.from_dict(data)
        
        return None
    
    def _update_existing_template(self, template: ReflexTemplate, 
                                 cycles: List[ReflexCycle]) -> ReflexTemplate:
        """Update existing template with new cycles."""
        # Add new source cycles
        new_cycle_ids = [c.cycle_id for c in cycles]
        template.source_cycles = list(set(template.source_cycles + new_cycle_ids))
        
        # Recalculate metrics
        all_cycles = self._get_cycles_by_ids(template.source_cycles)
        if all_cycles:
            template.success_rate = np.mean([1.0 if c.success else 0.0 for c in all_cycles])
            template.avg_score = np.mean([c.reflex_score.score if c.reflex_score else 0.0 for c in all_cycles])
            template.cluster_overlap = self._calculate_cluster_overlap(all_cycles)
            template.effectiveness_notes = self._generate_effectiveness_notes(all_cycles)
        
        # Update in database
        self._update_template(template)
        
        return template
    
    def _get_cycles_by_ids(self, cycle_ids: List[str]) -> List[ReflexCycle]:
        """Get reflex cycles by their IDs."""
        # This would need to be implemented based on how cycles are stored
        # For now, return empty list
        return []
    
    def _store_template(self, template: ReflexTemplate):
        """Store template in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO reflex_templates 
            (template_id, strategy, pattern_signature, goal_pattern, execution_pattern,
             success_rate, avg_score, cluster_overlap, usage_count, created_at, 
             last_used, source_cycles, effectiveness_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            template.template_id, template.strategy, template.pattern_signature,
            template.goal_pattern, json.dumps(template.execution_pattern),
            template.success_rate, template.avg_score, template.cluster_overlap,
            template.usage_count, template.created_at, template.last_used,
            json.dumps(template.source_cycles), template.effectiveness_notes
        ))
        
        conn.commit()
        conn.close()
    
    def _update_template(self, template: ReflexTemplate):
        """Update template in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE reflex_templates 
            SET goal_pattern = ?, execution_pattern = ?, success_rate = ?,
                avg_score = ?, cluster_overlap = ?, usage_count = ?, 
                last_used = ?, source_cycles = ?, effectiveness_notes = ?
            WHERE template_id = ?
        """, (
            template.goal_pattern, json.dumps(template.execution_pattern),
            template.success_rate, template.avg_score, template.cluster_overlap,
            template.usage_count, template.last_used, json.dumps(template.source_cycles),
            template.effectiveness_notes, template.template_id
        ))
        
        conn.commit()
        conn.close()
    
    def suggest_templates_for_drift(self, drift_score: float, token_id: Optional[int] = None,
                                   cluster_id: Optional[str] = None) -> List[ReflexTemplate]:
        """
        Suggest reflex templates for a given drift scenario.
        
        Args:
            drift_score: The drift score of the current scenario
            token_id: Optional token ID for more specific matching
            cluster_id: Optional cluster ID for more specific matching
            
        Returns:
            List of suggested ReflexTemplate objects
        """
        # Get all templates
        templates = self.get_all_templates()
        
        # Score templates for relevance
        scored_templates = []
        for template in templates:
            score = self._calculate_template_relevance(template, drift_score, token_id, cluster_id)
            scored_templates.append((template, score))
        
        # Sort by relevance score and return top suggestions
        scored_templates.sort(key=lambda x: x[1], reverse=True)
        return [template for template, score in scored_templates[:5]]
    
    def _calculate_template_relevance(self, template: ReflexTemplate, 
                                     drift_score: float, token_id: Optional[int] = None,
                                     cluster_id: Optional[str] = None) -> float:
        """Calculate relevance score for a template given current drift."""
        scores = []
        
        # Success rate weight
        scores.append(template.success_rate * 0.3)
        
        # Average score weight
        scores.append(template.avg_score * 0.2)
        
        # Usage count weight (normalized)
        usage_score = min(template.usage_count / 10.0, 1.0) * 0.1
        scores.append(usage_score)
        
        # Recency weight
        days_since_creation = (time.time() - template.created_at) / (24 * 3600)
        recency_score = max(0.0, 1.0 - (days_since_creation / 30.0)) * 0.1
        scores.append(recency_score)
        
        # Pattern matching weight
        pattern_score = self._calculate_pattern_match_score(template, drift_score, token_id, cluster_id)
        scores.append(pattern_score * 0.3)
        
        return sum(scores)
    
    def _calculate_pattern_match_score(self, template: ReflexTemplate, 
                                      drift_score: float, token_id: Optional[int] = None,
                                      cluster_id: Optional[str] = None) -> float:
        """Calculate how well a template matches the current drift pattern."""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated pattern matching
        
        # Drift score similarity
        if "high_severity" in template.goal_pattern and drift_score > 0.8:
            return 1.0
        elif "medium_severity" in template.goal_pattern and 0.5 <= drift_score <= 0.8:
            return 1.0
        elif "low_severity" in template.goal_pattern and drift_score < 0.5:
            return 1.0
        
        return 0.5
    
    def get_all_templates(self, limit: Optional[int] = None) -> List[ReflexTemplate]:
        """Get all templates, optionally limited."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT template_id, strategy, pattern_signature, goal_pattern,
                   execution_pattern, success_rate, avg_score, cluster_overlap,
                   usage_count, created_at, last_used, source_cycles, effectiveness_notes
            FROM reflex_templates 
            ORDER BY success_rate DESC, avg_score DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        
        templates = []
        for row in cursor.fetchall():
            data = {
                'template_id': row[0],
                'strategy': row[1],
                'pattern_signature': row[2],
                'goal_pattern': row[3],
                'execution_pattern': json.loads(row[4]),
                'success_rate': row[5],
                'avg_score': row[6],
                'cluster_overlap': row[7],
                'usage_count': row[8],
                'created_at': row[9],
                'last_used': row[10],
                'source_cycles': json.loads(row[11]),
                'effectiveness_notes': row[12]
            }
            templates.append(ReflexTemplate.from_dict(data))
        
        conn.close()
        return templates
    
    def get_templates_by_strategy(self, strategy: str, limit: Optional[int] = None) -> List[ReflexTemplate]:
        """Get templates for a specific strategy."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT template_id, strategy, pattern_signature, goal_pattern,
                   execution_pattern, success_rate, avg_score, cluster_overlap,
                   usage_count, created_at, last_used, source_cycles, effectiveness_notes
            FROM reflex_templates 
            WHERE strategy = ?
            ORDER BY success_rate DESC, avg_score DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, (strategy,))
        
        templates = []
        for row in cursor.fetchall():
            data = {
                'template_id': row[0],
                'strategy': row[1],
                'pattern_signature': row[2],
                'goal_pattern': row[3],
                'execution_pattern': json.loads(row[4]),
                'success_rate': row[5],
                'avg_score': row[6],
                'cluster_overlap': row[7],
                'usage_count': row[8],
                'created_at': row[9],
                'last_used': row[10],
                'source_cycles': json.loads(row[11]),
                'effectiveness_notes': row[12]
            }
            templates.append(ReflexTemplate.from_dict(data))
        
        conn.close()
        return templates
    
    def increment_usage(self, template_id: str):
        """Increment usage count for a template."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE reflex_templates 
            SET usage_count = usage_count + 1, last_used = ?
            WHERE template_id = ?
        """, (time.time(), template_id))
        
        conn.commit()
        conn.close()
    
    def get_template_statistics(self) -> Dict[str, Any]:
        """Get statistics about reflex templates."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total templates
        cursor.execute("SELECT COUNT(*) FROM reflex_templates")
        total_templates = cursor.fetchone()[0]
        
        # Average success rate
        cursor.execute("SELECT AVG(success_rate) FROM reflex_templates")
        avg_success_rate = cursor.fetchone()[0] or 0.0
        
        # Average score
        cursor.execute("SELECT AVG(avg_score) FROM reflex_templates")
        avg_score = cursor.fetchone()[0] or 0.0
        
        # Templates by strategy
        cursor.execute("""
            SELECT strategy, COUNT(*) 
            FROM reflex_templates 
            GROUP BY strategy 
            ORDER BY COUNT(*) DESC
        """)
        by_strategy = cursor.fetchall()
        
        # Most used templates
        cursor.execute("""
            SELECT template_id, goal_pattern, usage_count 
            FROM reflex_templates 
            ORDER BY usage_count DESC 
            LIMIT 5
        """)
        top_used = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_templates': total_templates,
            'average_success_rate': avg_success_rate,
            'average_score': avg_score,
            'templates_by_strategy': [
                {'strategy': row[0], 'count': row[1]}
                for row in by_strategy
            ],
            'top_used_templates': [
                {'template_id': row[0], 'goal_pattern': row[1], 'usage': row[2]}
                for row in top_used
            ]
        } 