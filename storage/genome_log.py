#!/usr/bin/env python3
"""
Genome Storage and Management for MeRNSTA Phase 20: Neural Evolution Tree & Genetic Self-Replication

Handles genome storage, lineage tracking, and genetic evolution persistence.
Each genome represents a complete state/version of the MeRNSTA system with mutations.
"""

import sqlite3
import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path


class GenomeStatus(Enum):
    """Status of a genome in the evolution tree"""
    ACTIVE = "active"          # Currently running/being tested
    ARCHIVED = "archived"      # Stored but not active
    FAILED = "failed"          # Failed performance tests
    ELITE = "elite"            # High-performing, promoted
    EXPERIMENTAL = "experimental"  # Under testing
    DEPRECATED = "deprecated"  # Superseded by better version


class MutationType(Enum):
    """Types of mutations that can occur"""
    CODE_EVOLUTION = "code_evolution"        # Code modifications
    CONFIG_UPDATE = "config_update"          # Configuration changes
    MEMORY_PRUNING = "memory_pruning"        # Memory optimization
    GOAL_MODIFICATION = "goal_modification"  # Goal/objective changes
    AGENT_UPDATE = "agent_update"            # Agent behavior changes
    ARCHITECTURE_CHANGE = "architecture_change"  # System architecture
    PERFORMANCE_TUNING = "performance_tuning"    # Performance optimizations
    CONSTRAINT_EVOLUTION = "constraint_evolution"  # Ethical constraint updates


@dataclass
class Mutation:
    """Represents a single mutation in genome evolution"""
    
    mutation_id: str
    mutation_type: MutationType
    description: str
    target_component: str  # Which part of system was modified
    changes: Dict[str, Any]  # Specific changes made
    timestamp: float = field(default_factory=time.time)
    success_rate: Optional[float] = None
    side_effects: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.mutation_type, str):
            self.mutation_type = MutationType(self.mutation_type)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert mutation to dictionary for storage"""
        data = asdict(self)
        data['mutation_type'] = self.mutation_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Mutation':
        """Create mutation from dictionary"""
        return cls(**data)


@dataclass
class Genome:
    """Represents a complete genome (version) of the MeRNSTA system"""
    
    genome_id: str
    parent_id: Optional[str] = None
    mutations: List[Mutation] = field(default_factory=list)
    fitness_score: float = 0.5  # Start with neutral fitness
    origin_timestamp: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    status: GenomeStatus = GenomeStatus.EXPERIMENTAL
    
    # Performance metrics
    success_rate: float = 0.0
    memory_efficiency: float = 0.0
    stability_score: float = 0.0
    response_quality: float = 0.0
    constraint_compliance: float = 1.0
    
    # System state snapshot
    config_hash: Optional[str] = None
    agent_versions: Dict[str, str] = field(default_factory=dict)
    memory_state_hash: Optional[str] = None
    
    # Lineage information
    generation: int = 0
    branch_name: Optional[str] = None
    descendant_count: int = 0
    
    # Metadata
    creator: str = "system"
    notes: str = ""
    test_results: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = GenomeStatus(self.status)
        
        # Convert mutation dicts to Mutation objects if needed
        converted_mutations = []
        for mutation in self.mutations:
            if isinstance(mutation, dict):
                converted_mutations.append(Mutation.from_dict(mutation))
            else:
                converted_mutations.append(mutation)
        self.mutations = converted_mutations
    
    def calculate_fitness(self) -> float:
        """Calculate overall fitness score from component metrics"""
        
        # Weighted combination of performance metrics
        weights = {
            'success_rate': 0.3,
            'memory_efficiency': 0.2,
            'stability_score': 0.2,
            'response_quality': 0.2,
            'constraint_compliance': 0.1
        }
        
        fitness = 0.0
        for metric, weight in weights.items():
            value = getattr(self, metric, 0.0)
            fitness += value * weight
        
        # Penalty for failed mutations
        failed_mutations = len([m for m in self.mutations if m.success_rate is not None and m.success_rate < 0.3])
        fitness -= failed_mutations * 0.05
        
        # Bonus for beneficial mutations
        beneficial_mutations = len([m for m in self.mutations if m.success_rate is not None and m.success_rate > 0.8])
        fitness += beneficial_mutations * 0.02
        
        self.fitness_score = max(0.0, min(1.0, fitness))
        return self.fitness_score
    
    def add_mutation(self, mutation: Mutation):
        """Add a mutation to this genome"""
        self.mutations.append(mutation)
        self.last_update = time.time()
        
        # Recalculate fitness
        self.calculate_fitness()
    
    def get_lineage_depth(self) -> int:
        """Get depth in evolution tree (generation number)"""
        return self.generation
    
    def is_descendant_of(self, ancestor_id: str, genome_log: 'GenomeLog') -> bool:
        """Check if this genome descends from the given ancestor"""
        current_id = self.parent_id
        
        while current_id:
            if current_id == ancestor_id:
                return True
            
            parent = genome_log.get_genome(current_id)
            if not parent:
                break
            current_id = parent.parent_id
        
        return False
    
    def get_mutation_summary(self) -> Dict[str, int]:
        """Get summary of mutation types in this genome"""
        summary = {}
        for mutation in self.mutations:
            mut_type = mutation.mutation_type.value
            summary[mut_type] = summary.get(mut_type, 0) + 1
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary for storage"""
        data = asdict(self)
        data['status'] = self.status.value
        data['mutations'] = [m.to_dict() for m in self.mutations]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Genome':
        """Create genome from dictionary"""
        return cls(**data)
    
    @classmethod
    def create_genesis_genome(cls, creator: str = "system") -> 'Genome':
        """Create the first genome (genesis) with no parent"""
        
        genome_id = cls._generate_genome_id("genesis")
        
        return cls(
            genome_id=genome_id,
            parent_id=None,
            generation=0,
            branch_name="genesis",
            creator=creator,
            status=GenomeStatus.ACTIVE,
            fitness_score=0.5,
            tags=["genesis", "original"],
            notes="Original genesis genome - base version of MeRNSTA system"
        )
    
    @classmethod
    def fork_from_parent(cls, parent: 'Genome', mutations: List[Mutation], 
                        creator: str = "system", branch_name: Optional[str] = None) -> 'Genome':
        """Create a new genome by forking from a parent with mutations"""
        
        # Generate ID based on parent and mutations
        mutation_summary = "_".join([m.mutation_type.value for m in mutations])
        genome_id = cls._generate_genome_id(f"{parent.genome_id}_{mutation_summary}")
        
        # Create new genome
        new_genome = cls(
            genome_id=genome_id,
            parent_id=parent.genome_id,
            mutations=mutations.copy(),
            generation=parent.generation + 1,
            branch_name=branch_name or f"branch_{genome_id[:8]}",
            creator=creator,
            status=GenomeStatus.EXPERIMENTAL,
            fitness_score=parent.fitness_score,  # Inherit parent's fitness initially
            tags=["forked"] + [m.mutation_type.value for m in mutations],
            
            # Inherit performance metrics (will be updated through testing)
            success_rate=parent.success_rate,
            memory_efficiency=parent.memory_efficiency,
            stability_score=parent.stability_score,
            response_quality=parent.response_quality,
            constraint_compliance=parent.constraint_compliance,
            
            # Copy system state
            config_hash=parent.config_hash,
            agent_versions=parent.agent_versions.copy(),
            memory_state_hash=parent.memory_state_hash
        )
        
        return new_genome
    
    @staticmethod
    def _generate_genome_id(base_string: str) -> str:
        """Generate unique genome ID"""
        timestamp = str(time.time())
        combined = f"{base_string}_{timestamp}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]


class GenomeLog:
    """Manages storage and retrieval of genomes with SQLite persistence"""
    
    def __init__(self, db_path: str = "genome_evolution.db"):
        self.db_path = db_path
        self.genomes: Dict[str, Genome] = {}
        self._init_database()
        self._load_genomes()
        
        logging.info(f"[GenomeLog] Initialized with {len(self.genomes)} genomes from {db_path}")
    
    def _init_database(self):
        """Initialize SQLite database for genome storage"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Genomes table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS genomes (
                genome_id TEXT PRIMARY KEY,
                parent_id TEXT,
                fitness_score REAL,
                origin_timestamp REAL,
                last_update REAL,
                status TEXT,
                generation INTEGER,
                branch_name TEXT,
                creator TEXT,
                notes TEXT,
                
                -- Performance metrics
                success_rate REAL,
                memory_efficiency REAL,
                stability_score REAL,
                response_quality REAL,
                constraint_compliance REAL,
                
                -- System state
                config_hash TEXT,
                memory_state_hash TEXT,
                
                -- Metadata
                descendant_count INTEGER,
                tags TEXT,  -- JSON array
                agent_versions TEXT,  -- JSON object
                test_results TEXT,  -- JSON object
                
                FOREIGN KEY (parent_id) REFERENCES genomes (genome_id)
            )
        ''')
        
        # Mutations table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS mutations (
                mutation_id TEXT PRIMARY KEY,
                genome_id TEXT,
                mutation_type TEXT,
                description TEXT,
                target_component TEXT,
                changes TEXT,  -- JSON object
                timestamp REAL,
                success_rate REAL,
                side_effects TEXT,  -- JSON array
                
                FOREIGN KEY (genome_id) REFERENCES genomes (genome_id)
            )
        ''')
        
        # Evolution events table (for tracking lineage events)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS evolution_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT,  -- fork, score_update, status_change, etc.
                genome_id TEXT,
                parent_id TEXT,
                timestamp REAL,
                details TEXT,  -- JSON object
                
                FOREIGN KEY (genome_id) REFERENCES genomes (genome_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_genomes(self):
        """Load all genomes from database"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Load genomes
        cursor = conn.execute('''
            SELECT * FROM genomes ORDER BY origin_timestamp
        ''')
        
        for row in cursor.fetchall():
            genome_data = {
                'genome_id': row[0],
                'parent_id': row[1],
                'fitness_score': row[2],
                'origin_timestamp': row[3],
                'last_update': row[4],
                'status': row[5],
                'generation': row[6],
                'branch_name': row[7],
                'creator': row[8],
                'notes': row[9],
                'success_rate': row[10],
                'memory_efficiency': row[11],
                'stability_score': row[12],
                'response_quality': row[13],
                'constraint_compliance': row[14],
                'config_hash': row[15],
                'memory_state_hash': row[16],
                'descendant_count': row[17],
                'tags': json.loads(row[18]) if row[18] else [],
                'agent_versions': json.loads(row[19]) if row[19] else {},
                'test_results': json.loads(row[20]) if row[20] else {},
                'mutations': []  # Will be loaded separately
            }
            
            genome = Genome.from_dict(genome_data)
            self.genomes[genome.genome_id] = genome
        
        # Load mutations for each genome
        for genome_id in self.genomes:
            cursor = conn.execute('''
                SELECT * FROM mutations WHERE genome_id = ? ORDER BY timestamp
            ''', (genome_id,))
            
            mutations = []
            for row in cursor.fetchall():
                mutation_data = {
                    'mutation_id': row[0],
                    'mutation_type': row[2],
                    'description': row[3],
                    'target_component': row[4],
                    'changes': json.loads(row[5]) if row[5] else {},
                    'timestamp': row[6],
                    'success_rate': row[7],
                    'side_effects': json.loads(row[8]) if row[8] else []
                }
                mutations.append(Mutation.from_dict(mutation_data))
            
            self.genomes[genome_id].mutations = mutations
        
        conn.close()
    
    def save_genome(self, genome: Genome):
        """Save genome to database"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Save genome
        conn.execute('''
            INSERT OR REPLACE INTO genomes 
            (genome_id, parent_id, fitness_score, origin_timestamp, last_update, status,
             generation, branch_name, creator, notes, success_rate, memory_efficiency,
             stability_score, response_quality, constraint_compliance, config_hash,
             memory_state_hash, descendant_count, tags, agent_versions, test_results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            genome.genome_id, genome.parent_id, genome.fitness_score,
            genome.origin_timestamp, genome.last_update, genome.status.value,
            genome.generation, genome.branch_name, genome.creator, genome.notes,
            genome.success_rate, genome.memory_efficiency, genome.stability_score,
            genome.response_quality, genome.constraint_compliance, genome.config_hash,
            genome.memory_state_hash, genome.descendant_count,
            json.dumps(genome.tags), json.dumps(genome.agent_versions),
            json.dumps(genome.test_results)
        ))
        
        # Save mutations
        for mutation in genome.mutations:
            conn.execute('''
                INSERT OR REPLACE INTO mutations
                (mutation_id, genome_id, mutation_type, description, target_component,
                 changes, timestamp, success_rate, side_effects)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                mutation.mutation_id, genome.genome_id, mutation.mutation_type.value,
                mutation.description, mutation.target_component,
                json.dumps(mutation.changes), mutation.timestamp,
                mutation.success_rate, json.dumps(mutation.side_effects)
            ))
        
        conn.commit()
        conn.close()
        
        # Update in-memory storage
        self.genomes[genome.genome_id] = genome
    
    def get_genome(self, genome_id: str) -> Optional[Genome]:
        """Get genome by ID"""
        return self.genomes.get(genome_id)
    
    def get_all_genomes(self) -> List[Genome]:
        """Get all genomes"""
        return list(self.genomes.values())
    
    def get_genomes_by_status(self, status: GenomeStatus) -> List[Genome]:
        """Get genomes with specific status"""
        return [g for g in self.genomes.values() if g.status == status]
    
    def get_children(self, parent_id: str) -> List[Genome]:
        """Get direct children of a genome"""
        return [g for g in self.genomes.values() if g.parent_id == parent_id]
    
    def get_descendants(self, ancestor_id: str) -> List[Genome]:
        """Get all descendants of a genome"""
        descendants = []
        for genome in self.genomes.values():
            if genome.is_descendant_of(ancestor_id, self):
                descendants.append(genome)
        return descendants
    
    def get_lineage_path(self, genome_id: str) -> List[Genome]:
        """Get full lineage path from genesis to genome"""
        path = []
        current = self.get_genome(genome_id)
        
        while current:
            path.insert(0, current)
            if current.parent_id:
                current = self.get_genome(current.parent_id)
            else:
                break
        
        return path
    
    def get_elite_genomes(self, threshold: float = 0.8) -> List[Genome]:
        """Get genomes with high fitness scores"""
        return [g for g in self.genomes.values() 
                if g.fitness_score >= threshold and g.status != GenomeStatus.FAILED]
    
    def get_failed_genomes(self, threshold: float = 0.2) -> List[Genome]:
        """Get genomes with low fitness scores"""
        return [g for g in self.genomes.values() 
                if g.fitness_score <= threshold or g.status == GenomeStatus.FAILED]
    
    def update_fitness(self, genome_id: str, fitness_score: float, 
                      metrics: Optional[Dict[str, float]] = None):
        """Update fitness score and metrics for a genome"""
        
        genome = self.get_genome(genome_id)
        if not genome:
            return False
        
        genome.fitness_score = fitness_score
        genome.last_update = time.time()
        
        if metrics:
            for metric, value in metrics.items():
                if hasattr(genome, metric):
                    setattr(genome, metric, value)
        
        # Recalculate fitness with new metrics
        genome.calculate_fitness()
        
        # Save to database
        self.save_genome(genome)
        
        logging.info(f"[GenomeLog] Updated fitness for {genome_id}: {fitness_score:.3f}")
        return True
    
    def archive_genome(self, genome_id: str, reason: str = ""):
        """Archive a genome"""
        genome = self.get_genome(genome_id)
        if genome:
            genome.status = GenomeStatus.ARCHIVED
            genome.notes += f"\nArchived: {reason}" if reason else "\nArchived"
            self.save_genome(genome)
            logging.info(f"[GenomeLog] Archived genome {genome_id}: {reason}")
    
    def mark_elite(self, genome_id: str, reason: str = ""):
        """Mark a genome as elite"""
        genome = self.get_genome(genome_id)
        if genome:
            genome.status = GenomeStatus.ELITE
            genome.notes += f"\nPromoted to elite: {reason}" if reason else "\nPromoted to elite"
            self.save_genome(genome)
            logging.info(f"[GenomeLog] Marked genome {genome_id} as elite: {reason}")
    
    def log_evolution_event(self, event_type: str, genome_id: str, 
                           parent_id: Optional[str] = None, details: Optional[Dict] = None):
        """Log an evolution event"""
        
        event_id = hashlib.sha256(f"{event_type}_{genome_id}_{time.time()}".encode()).hexdigest()[:16]
        
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO evolution_events
            (event_id, event_type, genome_id, parent_id, timestamp, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            event_id, event_type, genome_id, parent_id,
            time.time(), json.dumps(details) if details else None
        ))
        conn.commit()
        conn.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get evolution statistics"""
        
        genomes = list(self.genomes.values())
        
        if not genomes:
            return {'total_genomes': 0}
        
        status_counts = {}
        for status in GenomeStatus:
            status_counts[status.value] = len([g for g in genomes if g.status == status])
        
        fitness_scores = [g.fitness_score for g in genomes]
        
        # Generation statistics
        generations = [g.generation for g in genomes]
        max_generation = max(generations) if generations else 0
        
        # Mutation statistics
        mutation_counts = {}
        for genome in genomes:
            for mutation in genome.mutations:
                mut_type = mutation.mutation_type.value
                mutation_counts[mut_type] = mutation_counts.get(mut_type, 0) + 1
        
        return {
            'total_genomes': len(genomes),
            'status_distribution': status_counts,
            'fitness_stats': {
                'average': sum(fitness_scores) / len(fitness_scores),
                'max': max(fitness_scores),
                'min': min(fitness_scores)
            },
            'max_generation': max_generation,
            'mutation_types': mutation_counts,
            'active_branches': len(set(g.branch_name for g in genomes if g.branch_name)),
            'elite_count': len([g for g in genomes if g.status == GenomeStatus.ELITE])
        }


# Global genome log instance
_genome_log_instance = None

def get_genome_log() -> GenomeLog:
    """Get global genome log instance"""
    global _genome_log_instance
    if _genome_log_instance is None:
        _genome_log_instance = GenomeLog()
    return _genome_log_instance