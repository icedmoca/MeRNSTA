#!/usr/bin/env python3
"""
Test Suite for MeRNSTA Phase 20: Neural Evolution Tree & Genetic Self-Replication

Tests genome management, evolution tree operations, self-replication logic,
mutation tracking, fitness scoring, and CLI command integration.
"""

import pytest
import tempfile
import os
import sys
import time
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.genome_log import (
    Genome, GenomeLog, Mutation, MutationType, GenomeStatus,
    get_genome_log
)
from agents.evolution_tree import EvolutionTree, SelfReplicator


class TestGenome:
    """Test Genome class functionality"""
    
    def test_genome_creation(self):
        """Test basic genome creation"""
        genome = Genome(
            genome_id="test_genome_001",
            fitness_score=0.7,
            generation=1
        )
        
        assert genome.genome_id == "test_genome_001"
        assert genome.fitness_score == 0.7
        assert genome.generation == 1
        assert genome.status == GenomeStatus.EXPERIMENTAL
        assert len(genome.mutations) == 0
    
    def test_genesis_genome_creation(self):
        """Test genesis genome creation"""
        genesis = Genome.create_genesis_genome()
        
        assert genesis.parent_id is None
        assert genesis.generation == 0
        assert genesis.status == GenomeStatus.ACTIVE
        assert "genesis" in genesis.tags
        assert genesis.branch_name == "genesis"
    
    def test_fork_from_parent(self):
        """Test forking genome from parent"""
        parent = Genome.create_genesis_genome()
        
        mutation = Mutation(
            mutation_id=str(uuid.uuid4()),
            mutation_type=MutationType.CONFIG_UPDATE,
            description="Test mutation",
            target_component="config",
            changes={"test": "value"}
        )
        
        child = Genome.fork_from_parent(
            parent=parent,
            mutations=[mutation],
            creator="test",
            branch_name="test_branch"
        )
        
        assert child.parent_id == parent.genome_id
        assert child.generation == parent.generation + 1
        assert len(child.mutations) == 1
        assert child.mutations[0].description == "Test mutation"
        assert child.branch_name == "test_branch"
        assert child.status == GenomeStatus.EXPERIMENTAL
    
    def test_fitness_calculation(self):
        """Test fitness score calculation"""
        genome = Genome(
            genome_id="test_fitness",
            success_rate=0.8,
            memory_efficiency=0.7,
            stability_score=0.9,
            response_quality=0.85,
            constraint_compliance=1.0
        )
        
        fitness = genome.calculate_fitness()
        
        assert 0.0 <= fitness <= 1.0
        assert genome.fitness_score == fitness
        assert fitness > 0.5  # Should be good fitness
    
    def test_add_mutation(self):
        """Test adding mutations to genome"""
        genome = Genome(genome_id="test_mutation")
        
        mutation = Mutation(
            mutation_id=str(uuid.uuid4()),
            mutation_type=MutationType.PERFORMANCE_TUNING,
            description="Performance improvement",
            target_component="response_system",
            changes={"optimization": "enabled"}
        )
        
        initial_time = genome.last_update
        genome.add_mutation(mutation)
        
        assert len(genome.mutations) == 1
        assert genome.mutations[0] == mutation
        assert genome.last_update > initial_time
    
    def test_mutation_summary(self):
        """Test mutation type summary"""
        genome = Genome(genome_id="test_summary")
        
        mutations = [
            Mutation(str(uuid.uuid4()), MutationType.CONFIG_UPDATE, "config", "config", {}),
            Mutation(str(uuid.uuid4()), MutationType.CONFIG_UPDATE, "config2", "config", {}),
            Mutation(str(uuid.uuid4()), MutationType.MEMORY_PRUNING, "memory", "memory", {})
        ]
        
        for mutation in mutations:
            genome.add_mutation(mutation)
        
        summary = genome.get_mutation_summary()
        
        assert summary[MutationType.CONFIG_UPDATE.value] == 2
        assert summary[MutationType.MEMORY_PRUNING.value] == 1
    
    def test_genome_serialization(self):
        """Test genome to/from dict conversion"""
        original = Genome(
            genome_id="test_serial",
            fitness_score=0.8,
            tags=["test", "serialization"],
            notes="Test genome"
        )
        
        # Add a mutation
        mutation = Mutation(
            str(uuid.uuid4()), MutationType.CODE_EVOLUTION, 
            "test", "test", {"key": "value"}
        )
        original.add_mutation(mutation)
        
        # Convert to dict and back
        genome_dict = original.to_dict()
        restored = Genome.from_dict(genome_dict)
        
        assert restored.genome_id == original.genome_id
        assert restored.fitness_score == original.fitness_score
        assert restored.tags == original.tags
        assert restored.notes == original.notes
        assert len(restored.mutations) == len(original.mutations)
        assert restored.mutations[0].description == original.mutations[0].description


class TestMutation:
    """Test Mutation class functionality"""
    
    def test_mutation_creation(self):
        """Test basic mutation creation"""
        mutation = Mutation(
            mutation_id="test_mut_001",
            mutation_type=MutationType.CODE_EVOLUTION,
            description="Test code evolution",
            target_component="agents/test.py",
            changes={"function": "improved", "efficiency": "+20%"}
        )
        
        assert mutation.mutation_id == "test_mut_001"
        assert mutation.mutation_type == MutationType.CODE_EVOLUTION
        assert mutation.description == "Test code evolution"
        assert mutation.target_component == "agents/test.py"
        assert mutation.changes["efficiency"] == "+20%"
    
    def test_mutation_serialization(self):
        """Test mutation to/from dict conversion"""
        original = Mutation(
            mutation_id="serial_test",
            mutation_type=MutationType.MEMORY_PRUNING,
            description="Memory optimization",
            target_component="memory_system",
            changes={"pruning_rate": 0.1, "compression": True},
            side_effects=["faster_response", "reduced_capacity"]
        )
        
        # Convert to dict and back
        mutation_dict = original.to_dict()
        restored = Mutation.from_dict(mutation_dict)
        
        assert restored.mutation_id == original.mutation_id
        assert restored.mutation_type == original.mutation_type
        assert restored.description == original.description
        assert restored.changes == original.changes
        assert restored.side_effects == original.side_effects
    
    def test_mutation_type_conversion(self):
        """Test string to enum conversion for mutation types"""
        mutation = Mutation(
            mutation_id="enum_test",
            mutation_type="performance_tuning",  # String instead of enum
            description="Test",
            target_component="test",
            changes={}
        )
        
        assert isinstance(mutation.mutation_type, MutationType)
        assert mutation.mutation_type == MutationType.PERFORMANCE_TUNING


class TestGenomeLog:
    """Test GenomeLog class functionality"""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_genome_log_initialization(self, temp_db_path):
        """Test genome log initialization"""
        genome_log = GenomeLog(temp_db_path)
        
        assert genome_log.db_path == temp_db_path
        assert isinstance(genome_log.genomes, dict)
        assert os.path.exists(temp_db_path)
    
    def test_save_and_load_genome(self, temp_db_path):
        """Test saving and loading genomes"""
        genome_log = GenomeLog(temp_db_path)
        
        # Create and save genome
        genome = Genome.create_genesis_genome()
        genome_log.save_genome(genome)
        
        # Retrieve genome
        retrieved = genome_log.get_genome(genome.genome_id)
        
        assert retrieved is not None
        assert retrieved.genome_id == genome.genome_id
        assert retrieved.status == genome.status
        assert retrieved.generation == genome.generation
    
    def test_genome_with_mutations_persistence(self, temp_db_path):
        """Test persistence of genomes with mutations"""
        genome_log = GenomeLog(temp_db_path)
        
        # Create genome with mutations
        parent = Genome.create_genesis_genome()
        mutation = Mutation(
            str(uuid.uuid4()), MutationType.CONFIG_UPDATE,
            "Test mutation", "config", {"test": True}
        )
        child = Genome.fork_from_parent(parent, [mutation])
        
        # Save both
        genome_log.save_genome(parent)
        genome_log.save_genome(child)
        
        # Create new genome log instance (simulates restart)
        new_genome_log = GenomeLog(temp_db_path)
        
        # Retrieve child
        retrieved_child = new_genome_log.get_genome(child.genome_id)
        
        assert retrieved_child is not None
        assert len(retrieved_child.mutations) == 1
        assert retrieved_child.mutations[0].description == "Test mutation"
        assert retrieved_child.parent_id == parent.genome_id
    
    def test_fitness_update(self, temp_db_path):
        """Test fitness score updates"""
        genome_log = GenomeLog(temp_db_path)
        
        genome = Genome.create_genesis_genome()
        genome_log.save_genome(genome)
        
        # Update fitness
        metrics = {
            'success_rate': 0.8,
            'memory_efficiency': 0.9,
            'stability_score': 0.7
        }
        
        success = genome_log.update_fitness(genome.genome_id, 0.85, metrics)
        
        assert success == True
        
        updated_genome = genome_log.get_genome(genome.genome_id)
        assert updated_genome.fitness_score == 0.85
        assert updated_genome.success_rate == 0.8
        assert updated_genome.memory_efficiency == 0.9
    
    def test_genome_filtering(self, temp_db_path):
        """Test genome filtering by status and fitness"""
        genome_log = GenomeLog(temp_db_path)
        
        # Create genomes with different statuses and fitness
        genomes = []
        for i in range(5):
            genome = Genome(
                genome_id=f"test_genome_{i}",
                fitness_score=i * 0.2,
                status=GenomeStatus.ACTIVE if i % 2 == 0 else GenomeStatus.ARCHIVED
            )
            genomes.append(genome)
            genome_log.save_genome(genome)
        
        # Test status filtering
        active_genomes = genome_log.get_genomes_by_status(GenomeStatus.ACTIVE)
        assert len(active_genomes) == 3  # 0, 2, 4
        
        # Test elite filtering
        elite_genomes = genome_log.get_elite_genomes(0.6)
        assert len(elite_genomes) == 2  # fitness 0.8 and 0.6
        
        # Test failed filtering
        failed_genomes = genome_log.get_failed_genomes(0.2)
        assert len(failed_genomes) == 2  # fitness 0.0 and 0.2
    
    def test_lineage_operations(self, temp_db_path):
        """Test lineage tracking operations"""
        genome_log = GenomeLog(temp_db_path)
        
        # Create lineage: parent -> child1 -> grandchild
        parent = Genome.create_genesis_genome()
        child1_mutation = Mutation(str(uuid.uuid4()), MutationType.CONFIG_UPDATE, "child1", "config", {})
        child1 = Genome.fork_from_parent(parent, [child1_mutation])
        
        grandchild_mutation = Mutation(str(uuid.uuid4()), MutationType.MEMORY_PRUNING, "grandchild", "memory", {})
        grandchild = Genome.fork_from_parent(child1, [grandchild_mutation])
        
        # Save all
        for genome in [parent, child1, grandchild]:
            genome_log.save_genome(genome)
        
        # Test children retrieval
        children = genome_log.get_children(parent.genome_id)
        assert len(children) == 1
        assert children[0].genome_id == child1.genome_id
        
        # Test descendants retrieval
        descendants = genome_log.get_descendants(parent.genome_id)
        assert len(descendants) == 2  # child1 and grandchild
        
        # Test lineage path
        lineage = genome_log.get_lineage_path(grandchild.genome_id)
        assert len(lineage) == 3
        assert lineage[0].genome_id == parent.genome_id
        assert lineage[1].genome_id == child1.genome_id
        assert lineage[2].genome_id == grandchild.genome_id
        
        # Test descendant check
        assert grandchild.is_descendant_of(parent.genome_id, genome_log) == True
        assert parent.is_descendant_of(grandchild.genome_id, genome_log) == False


class TestEvolutionTree:
    """Test EvolutionTree class functionality"""
    
    @pytest.fixture
    def temp_evolution_tree(self):
        """Create temporary evolution tree"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        genome_log = GenomeLog(temp_path)
        evolution_tree = EvolutionTree(genome_log)
        
        yield evolution_tree
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_evolution_tree_initialization(self, temp_evolution_tree):
        """Test evolution tree initialization"""
        tree = temp_evolution_tree
        
        assert tree.genome_log is not None
        assert tree.current_genome_id is not None
        assert tree.max_active_genomes > 0
        
        # Should have created genesis genome
        current_genome = tree.genome_log.get_genome(tree.current_genome_id)
        assert current_genome is not None
        assert current_genome.generation == 0
    
    def test_genome_forking(self, temp_evolution_tree):
        """Test genome forking functionality"""
        tree = temp_evolution_tree
        
        # Create mutation
        mutation = Mutation(
            str(uuid.uuid4()), MutationType.PERFORMANCE_TUNING,
            "Performance improvement", "system", {"optimization": True}
        )
        
        # Fork genome
        new_genome_id = tree.fork_genome(mutations=[mutation], creator="test")
        
        assert new_genome_id is not None
        
        new_genome = tree.genome_log.get_genome(new_genome_id)
        assert new_genome is not None
        assert new_genome.parent_id == tree.current_genome_id
        assert len(new_genome.mutations) == 1
        assert new_genome.mutations[0].description == "Performance improvement"
    
    def test_fitness_scoring(self, temp_evolution_tree):
        """Test genome fitness scoring"""
        tree = temp_evolution_tree
        
        # Create metrics
        metrics = {
            'success_rate': 0.9,
            'memory_efficiency': 0.8,
            'stability_score': 0.85,
            'response_quality': 0.9,
            'constraint_compliance': 1.0
        }
        
        # Score current genome
        success = tree.score_genome(tree.current_genome_id, metrics)
        
        assert success == True
        
        current_genome = tree.genome_log.get_genome(tree.current_genome_id)
        assert current_genome.fitness_score > 0.8  # Should be high
    
    def test_elite_selection(self, temp_evolution_tree):
        """Test elite genome selection"""
        tree = temp_evolution_tree
        
        # Create several genomes with different fitness
        genomes = []
        for i in range(3):
            mutation = Mutation(str(uuid.uuid4()), MutationType.CONFIG_UPDATE, f"test_{i}", "config", {})
            genome_id = tree.fork_genome(mutations=[mutation])
            
            # Set different fitness scores
            fitness = 0.5 + (i * 0.2)
            metrics = {'overall_fitness': fitness}
            tree.score_genome(genome_id, metrics)
            
            genomes.append(genome_id)
        
        # Select elite
        elite_genomes = tree.select_elite_branches(limit=2)
        
        # Should return genomes sorted by fitness
        assert len(elite_genomes) <= 2
        if len(elite_genomes) > 1:
            assert elite_genomes[0].fitness_score >= elite_genomes[1].fitness_score
    
    def test_genome_activation(self, temp_evolution_tree):
        """Test genome activation (switching)"""
        tree = temp_evolution_tree
        
        # Create new genome
        mutation = Mutation(str(uuid.uuid4()), MutationType.CODE_EVOLUTION, "new version", "code", {})
        new_genome_id = tree.fork_genome(mutations=[mutation])
        
        original_genome_id = tree.current_genome_id
        
        # Activate new genome
        success = tree.activate_genome(new_genome_id)
        
        assert success == True
        assert tree.current_genome_id == new_genome_id
        
        # Check status changes
        new_genome = tree.genome_log.get_genome(new_genome_id)
        assert new_genome.status == GenomeStatus.ACTIVE
    
    def test_lineage_visualization(self, temp_evolution_tree):
        """Test lineage visualization data generation"""
        tree = temp_evolution_tree
        
        # Create some lineage
        for i in range(3):
            mutation = Mutation(str(uuid.uuid4()), MutationType.CONFIG_UPDATE, f"gen_{i}", "config", {})
            tree.fork_genome(mutations=[mutation])
        
        # Generate visualization data
        lineage_data = tree.visualize_lineage()
        
        assert 'nodes' in lineage_data
        assert 'edges' in lineage_data
        assert 'root' in lineage_data
        assert len(lineage_data['nodes']) >= 2  # At least genesis + 1 fork


class TestSelfReplicator:
    """Test SelfReplicator agent functionality"""
    
    @pytest.fixture
    def temp_self_replicator(self):
        """Create temporary self-replicator"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        # Mock config to use temp database
        import agents.evolution_tree
        original_get_config = agents.evolution_tree.get_config
        
        def mock_get_config():
            return {
                'self_replication': {
                    'auto_replicate_on_drift': True,
                    'drift_threshold': 0.3,
                    'performance_window_hours': 1,
                    'min_replication_interval_hours': 0.1
                }
            }
        
        agents.evolution_tree.get_config = mock_get_config
        
        # Create instance
        replicator = SelfReplicator()
        replicator.evolution_tree.genome_log.db_path = temp_path
        
        yield replicator
        
        # Restore original config
        agents.evolution_tree.get_config = original_get_config
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_self_replicator_initialization(self, temp_self_replicator):
        """Test self-replicator initialization"""
        replicator = temp_self_replicator
        
        assert replicator.evolution_tree is not None
        assert replicator.genome_log is not None
        assert replicator.auto_replicate_enabled == True
        assert hasattr(replicator, 'performance_history')
    
    def test_performance_metrics_collection(self, temp_self_replicator):
        """Test performance metrics collection"""
        replicator = temp_self_replicator
        
        metrics = replicator._get_current_performance_metrics()
        
        assert 'success_rate' in metrics
        assert 'memory_efficiency' in metrics
        assert 'stability_score' in metrics
        assert 'response_quality' in metrics
        assert 'constraint_compliance' in metrics
        assert 'overall_fitness' in metrics
        
        # All metrics should be between 0 and 1
        for metric, value in metrics.items():
            assert 0.0 <= value <= 1.0
    
    def test_drift_detection(self, temp_self_replicator):
        """Test performance drift detection"""
        replicator = temp_self_replicator
        
        # Add performance history with declining trend
        declining_metrics = [
            {'overall_fitness': 0.8},
            {'overall_fitness': 0.7},
            {'overall_fitness': 0.6},
            {'overall_fitness': 0.4},  # Should trigger drift
            {'overall_fitness': 0.3}
        ]
        
        drift_detected = False
        for metrics in declining_metrics:
            drift_detected = replicator._detect_performance_drift(metrics)
        
        assert drift_detected == True
    
    def test_intelligent_mutation_generation(self, temp_self_replicator):
        """Test intelligent mutation generation"""
        replicator = temp_self_replicator
        
        # Simulate poor performance metrics
        poor_metrics = {
            'memory_efficiency': 0.3,  # Low memory efficiency
            'response_quality': 0.4,   # Low response quality
            'overall_fitness': 0.5
        }
        
        mutations = replicator._generate_intelligent_mutations(poor_metrics, "performance_test")
        
        assert len(mutations) > 0
        
        # Should include memory optimization
        memory_mutations = [m for m in mutations if m.mutation_type == MutationType.MEMORY_PRUNING]
        assert len(memory_mutations) > 0
        
        # Should include performance tuning
        performance_mutations = [m for m in mutations if m.mutation_type == MutationType.PERFORMANCE_TUNING]
        assert len(performance_mutations) > 0
    
    def test_agent_response_interface(self, temp_self_replicator):
        """Test agent response interface"""
        replicator = temp_self_replicator
        
        # Test statistics query
        response = replicator.respond("show statistics")
        assert 'response' in response
        assert 'statistics' in response
        assert 'agent' in response
        assert response['agent'] == 'self_replicator'
        
        # Test drift analysis
        response = replicator.respond("analyze drift")
        assert 'drift_detected' in response
        assert 'performance_metrics' in response
        
        # Test fitness evaluation
        response = replicator.respond("evaluate fitness")
        assert 'fitness_updated' in response or 'metrics' in response
    
    def test_autonomous_evolution_trigger(self, temp_self_replicator):
        """Test autonomous evolution triggering"""
        replicator = temp_self_replicator
        
        # Reset replication time to allow immediate replication
        replicator.last_replication_time = 0
        
        # Trigger evolution
        new_genome_id = replicator.trigger_autonomous_evolution("test_trigger")
        
        if new_genome_id:  # May be None if conditions not met
            assert new_genome_id is not None
            assert len(new_genome_id) > 0
            
            # Verify new genome exists
            new_genome = replicator.genome_log.get_genome(new_genome_id)
            assert new_genome is not None
            assert new_genome.creator == "autonomous_system"


class TestCLIIntegration:
    """Test CLI command integration"""
    
    def test_evolution_imports(self):
        """Test that evolution components can be imported for CLI"""
        try:
            from agents.evolution_tree import get_evolution_tree, get_self_replicator
            from storage.genome_log import Mutation, MutationType
            
            # Test instantiation
            tree = get_evolution_tree()
            replicator = get_self_replicator()
            
            assert tree is not None
            assert replicator is not None
            
        except ImportError as e:
            pytest.fail(f"Could not import evolution components for CLI: {e}")
    
    def test_mutation_creation_for_cli(self):
        """Test mutation creation as done in CLI commands"""
        import uuid
        from storage.genome_log import Mutation, MutationType
        
        mutation = Mutation(
            mutation_id=str(uuid.uuid4()),
            mutation_type=MutationType.CODE_EVOLUTION,
            description="CLI test mutation",
            target_component="cli_test",
            changes={"test": True}
        )
        
        assert mutation.mutation_id is not None
        assert mutation.mutation_type == MutationType.CODE_EVOLUTION
        assert mutation.description == "CLI test mutation"


class TestIntegration:
    """Test full integration scenarios"""
    
    def test_complete_evolution_workflow(self):
        """Test complete evolution workflow from genesis to elite"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create evolution system
            genome_log = GenomeLog(temp_path)
            evolution_tree = EvolutionTree(genome_log)
            
            # Verify genesis creation
            genesis_id = evolution_tree.current_genome_id
            genesis = genome_log.get_genome(genesis_id)
            assert genesis is not None
            assert genesis.generation == 0
            
            # Fork new genome
            mutation = Mutation(
                str(uuid.uuid4()), MutationType.PERFORMANCE_TUNING,
                "Integration test mutation", "system", {"test": True}
            )
            child_id = evolution_tree.fork_genome(mutations=[mutation])
            
            # Score the child highly
            high_metrics = {
                'success_rate': 0.95,
                'memory_efficiency': 0.9,
                'stability_score': 0.95,
                'response_quality': 0.9,
                'constraint_compliance': 1.0
            }
            evolution_tree.score_genome(child_id, high_metrics)
            
            # Verify elite promotion
            child = genome_log.get_genome(child_id)
            assert child.fitness_score > 0.8
            
            # Generate lineage visualization
            lineage = evolution_tree.visualize_lineage()
            assert len(lineage['nodes']) >= 2
            assert len(lineage['edges']) >= 1
            
            # Verify lineage integrity
            lineage_path = genome_log.get_lineage_path(child_id)
            assert len(lineage_path) == 2
            assert lineage_path[0].genome_id == genesis_id
            assert lineage_path[1].genome_id == child_id
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_multi_generation_evolution(self):
        """Test evolution across multiple generations"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        try:
            genome_log = GenomeLog(temp_path)
            evolution_tree = EvolutionTree(genome_log)
            
            # Create multiple generations
            current_id = evolution_tree.current_genome_id
            
            for generation in range(1, 4):
                mutation = Mutation(
                    str(uuid.uuid4()), MutationType.CONFIG_UPDATE,
                    f"Generation {generation} mutation", "config", {"gen": generation}
                )
                
                new_id = evolution_tree.fork_genome(
                    parent_id=current_id,
                    mutations=[mutation],
                    branch_name=f"gen_{generation}"
                )
                
                # Score this generation
                fitness = 0.5 + (generation * 0.1)
                metrics = {'overall_fitness': fitness}
                evolution_tree.score_genome(new_id, metrics)
                
                current_id = new_id
            
            # Verify genealogy
            final_genome = genome_log.get_genome(current_id)
            assert final_genome.generation == 3
            
            lineage_path = genome_log.get_lineage_path(current_id)
            assert len(lineage_path) == 4  # Genesis + 3 generations
            
            # Verify generation progression
            for i, genome in enumerate(lineage_path):
                assert genome.generation == i
            
            # Test descendant relationships
            genesis_id = lineage_path[0].genome_id
            descendants = genome_log.get_descendants(genesis_id)
            assert len(descendants) == 3  # 3 child generations
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])